# 19 · Worker 失败重试与状态落盘

> 关联代码：`app/core/job_failure.py`、`app/workers/index_worker.py`、`app/workers/protocol_pdf_worker.py`、`app/services/index_job_queue.py`、`app/services/protocol_pdf_job_queue.py`
> 关联测试：`tests/unit/test_job_failure.py`
> 前置阅读：[14 · 错误分类与降级原因体系](14-错误分类与降级原因体系.md)（`is_retryable` 的实现在那里）、[15 · request_id 全链路追踪](15-request_id全链路追踪.md)（`job.meta` 透传 rid）

---

## 1. 为什么要改

### 1.1 一个具体的翻车现场

用户上传了一份协议 PDF。接口返回 202，前端开始转圈。

后台 worker 拿到任务，跑到第 3 页时 pdfplumber 抛了 `ValueError: invalid font descriptor`。这是个确定性错误 —— 这份 PDF 就是坏的，重试一万次也是同样结果。理想的结果是：**任务标记 FAILED，错误信息写「PDF 第 3 页字体描述损坏」，页面上显示失败，用户重新导出一份 PDF 再上传。**

实际发生的是这样：

worker 的 `except` 块开始抢救，走到 `db.commit()` 这一行。但这个 `db` 是**同一个 session** —— 刚才那个异常已经让它进入了需要 rollback 的状态，而且更糟的情况是，如果原始异常本身就是数据库连接断开（这非常常见），那这次 commit 会抛出第二个异常：

```
OperationalError: server closed the connection unexpectedly
```

这个新异常从 `except` 块里逃出去，**顶掉了原始异常**。

Python 会在 traceback 里留一句 "During handling of the above exception, another exception occurred"。但你排障时看到的第一样东西不是完整 traceback，而是：

- RQ 存进 `job.exc_info` 的字符串
- 告警群里那条消息的标题
- `error_message` 字段（如果这次连它都没写进去）

这三个地方看到的全是 `OperationalError: server closed the connection`。于是你花两小时去查数据库连接池、查网络、查 PG 日志 —— 而真正的问题是**一份坏 PDF**。

### 1.2 更隐蔽的第二个后果

抢救失败意味着 `job.status = FAILED` 这句话没写进数据库。

于是：

- `protocol_ingestion_job` 那一行永远停在 `RUNNING`
- `protocol_ingestion` 那一行永远停在 `PROCESSING`
- 前端页面上永远显示「解析中」

用户盯着那个转圈的图标看了三天。没有任何人知道它其实第一分钟就死了。没有告警，因为告警是按 `status = FAILED` 触发的。

这是异步任务系统里最典型的 bug 类型：**任务卡在中间态**。它比「任务失败」危险得多 —— 失败会触发告警、会被看见、会被修；卡在中间态不会触发任何东西，只会消耗用户的耐心。

### 1.3 第三个后果：抖动等于终局

改之前的入队代码（`git show HEAD:app/services/index_job_queue.py`）：

```python
def enqueue_index_job(job_id: str) -> Job:
    return index_queue.enqueue(
        run_index_job,
        job_id,
        job_timeout="30m",
        failure_ttl=7 * 24 * 60 * 60,
        result_ttl=7 * 24 * 60 * 60,
    )
```

没有 `retry` 参数。RQ 的默认行为是**不重试** —— 任务抛异常一次，直接进 failed registry，终局。

于是：Milvus 正在选主（大概 3~8 秒），这期间入队的所有索引任务全部失败。八秒后集群恢复正常，但那些任务已经死了，需要人工重新触发。

一个几秒钟就自愈的抖动，代价是一批任务全废 + 人工介入。

### 1.4 第四个后果：无条件重试同样是浪费

但简单加个 `retry=Retry(max=3)` 也不对。RQ 的 `Retry` 是**无条件**的：只要任务抛异常就重排队，不区分异常类型。

于是那份坏 PDF 会被解析四次（首次 + 3 次重试），间隔 10、30、60 秒。四次一模一样的 `invalid font descriptor`。代价是：

- 白烧四倍 CPU 和 worker 槽位
- 最终失败被推迟了 100 秒 —— 用户多等一分半才知道要重新导出 PDF
- 日志里四条一样的错误，排障时以为是间歇性问题

所以需要的不是「要不要重试」这个开关，而是**按异常类型裁决**。

### 1.5 四个问题归纳

| 问题 | 后果 |
|---|---|
| 抢救用同一个 session | 二次异常顶掉原始异常，根因消失 |
| 抢救失败没兜底 | 任务永远卡在 RUNNING，无告警 |
| 完全没有重试 | 几秒的抖动 = 一批任务全废 |
| 无条件重试（若简单加上） | 确定性失败白烧四倍资源，延后暴露 |

---

## 2. 改之前什么样

### 2.1 `app/workers/index_worker.py` 的 except 块

从 git 里取出改造前的原样（`git show HEAD:app/workers/index_worker.py`）：

```python
    except Exception as exc:
        db.rollback()
        job = repo.get_index_job(job_id)
        if job is not None:
            job.status = IndexJobStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            document = repo.get_document(job.document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error_message = str(exc)
            db.commit()                                  # ← 炸点
        logger.exception(f"索引任务失败: job_id={job_id}")
        raise
    finally:
        db.close()                                       # ← 也可能炸
```

### 2.2 `app/workers/protocol_pdf_worker.py` 的 except 块

一模一样的形状（`git show HEAD:app/workers/protocol_pdf_worker.py:36-51`）：

```python
    except Exception as exc:
        db.rollback()
        job = repo.get_job(job_id)
        if job is not None:
            job.status = ProtocolIngestionJobStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            ingestion = repo.get_ingestion(job.ingestion_id)
            if ingestion is not None:
                ingestion.status = ProtocolIngestionStatus.FAILED
                ingestion.error_message = str(exc)
            db.commit()
        logger.exception(f"协议 PDF 入库任务失败: job_id={job_id}")
        raise
    finally:
        db.close()
```

两个文件里同一段逻辑抄了两遍。这本身就是个信号：**重复的代码意味着修 bug 要修两处，而且总有一处会被漏掉。**

### 2.3 这段代码里到底有几个炸点

逐行数一遍：

| 行 | 风险 |
|---|---|
| `db.rollback()` | 连接已断时抛 `OperationalError` —— 而且它是 except 块的第一行，一炸后面全跳过 |
| `repo.get_index_job(job_id)` | 同一个废 session 上的查询，抛 `PendingRollbackError` |
| `db.commit()` | 最经典的炸点，二次异常从这里逃出去 |
| `db.close()` | 在 `finally` 里，抛出来同样顶掉原始异常 |

四个炸点，全都在「原始异常已经发生之后」。而它们炸的条件和原始异常的成因高度相关 —— **越是基础设施故障，越会丢失真实原因，而那正是最需要看清原因的时候。**

---

## 3. 改之后什么样

新增一个模块 `app/core/job_failure.py` 承载全部失败处置逻辑，两个 worker 都调它。

### 3.1 模块职责说明（`app/core/job_failure.py:1-42`）

```python
"""RQ 任务失败处置：状态落盘 + 重试裁决。

## 处置

1. 用一个**独立的短生命周期 session** 写失败状态。原 session 的连接可能
   已经废了（连 rollback 都可能抛），在同一个连接上抢救等于在漏水的船上补漏。
2. 整段抢救逻辑包在 try 里，**任何二次失败只记日志，绝不外抛**。
   原始异常必须原样 raise 出去 —— 那才是要修的东西。
3. 重试裁决与「是否可重试」的判定复用 `app/core/errors.py` 的
   `is_retryable`，不在这里重写一遍分类逻辑（DRY）。
"""
```

第 3 条是刻意的。异常分类这件事在这个项目里有**唯一实现处**，就是 `app/core/errors.py:258` 的 `is_retryable`。worker 不自己 `isinstance` 一遍，HTTP 层也不自己判一遍。详见 [第 14 篇](14-错误分类与降级原因体系.md)。

### 3.2 重试额度（`app/core/job_failure.py:56-74`）

```python
# 默认重试策略：最多 3 次，间隔递增。
#
# 为什么间隔要递增而不是固定：瞬时故障（Milvus 选主、网络抖动、LLM 限流）
# 的恢复时间是不确定的。固定 10 秒重试三次，总共只覆盖 30 秒窗口；
# 递增到 60 秒能覆盖到分钟级抖动，而代价只是失败任务晚一点定性。
#
# 为什么是 3 次而不是更多：超过 3 次基本说明不是抖动而是真故障，
# 继续重试只是在推迟暴露问题，同时占着 worker 不干别的活。
_DEFAULT_RETRY_MAX = 3
_DEFAULT_RETRY_INTERVALS = [10, 30, 60]


def default_job_retry() -> Retry:
    """入队侧用的默认重试策略。

    定义成函数而不是模块级常量：`Retry` 实例携带可变状态，
    两个队列共享同一个对象是自找麻烦。
    """
    return Retry(max=_DEFAULT_RETRY_MAX, interval=_DEFAULT_RETRY_INTERVALS)
```

逐点拆开：

**为什么是函数不是常量。** 如果写成 `DEFAULT_RETRY = Retry(max=3, interval=[10,30,60])`，两个队列引用同一个对象。`Retry` 内部带可变状态，RQ 在处理时可能读写它。这类共享可变状态的 bug 表现为「A 队列的任务影响了 B 队列的重试次数」，极难复现。函数每次返回新实例，代价是一次对象分配，换来彻底的隔离。这条对应测试 `tests/unit/test_job_failure.py:151`：

```python
    def test_returns_fresh_instance_each_call(self) -> None:
        """Retry 实例带可变状态，两个队列共享同一个对象是自找麻烦。"""
        first, second = default_job_retry(), default_job_retry()

        assert first is not second
```

**为什么间隔递增。** 这是退避（backoff）的基本思路。假设 Milvus 选主要 8 秒：固定 10 秒间隔重试三次，覆盖的时间点是 10s / 20s / 30s，能撞上恢复；但如果是 PG 主从切换（通常 30~60 秒），10/20/30 三次全部落在故障窗口内，白试三次。递增到 10/40/100（累计），最后一次已经过了一分半，覆盖面大得多。

代价是：真正的失败任务要一分半后才定性。这个代价可以接受，因为失败任务多等一分钟没人受伤；而抖动导致的批量失败需要人工重跑，很痛。

### 3.3 重试裁决（`app/core/job_failure.py:77-128`）

```python
def apply_retry_policy(exc: BaseException, *, job_id: str) -> bool:
    """裁决这次失败是否值得让 RQ 重排队，并在不值得时就地否决。

    返回 True 表示 RQ 还会重试，False 表示这次失败是终局。
    """
    retryable = is_retryable(exc)                      # ① 复用唯一分类实现

    job = None
    with suppress(Exception):                          # ② 拿不到 job 不该影响任务
        job = get_current_job()

    if job is None:
        return retryable                               # ③ 脱离 RQ 上下文时保持纯函数

    retries_left = getattr(job, "retries_left", None) or 0

    if not retryable and retries_left > 0:
        # 清零而不是设成 None：`should_retry` 判的是 `is not None and > 0`，
        # 两种写法都能生效，但 0 更能表达「额度用完了」而非「没配过重试」。
        job.retries_left = 0                           # ④ 就地否决
        logger.warning(
            f"任务失败且不可重试，已取消剩余 {retries_left} 次重试: "
            f"job_id={job_id}, error_code={error_code_of(exc)}"
        )
        return False

    return retryable and retries_left > 0              # ⑤ 两个条件都要满足
```

① **`is_retryable` 不在这里重写。** 「哪些异常算瞬时故障」这个知识点在 `errors.py:222-255` 有唯一定义（两张 frozenset）。worker 只是消费者。

② **`get_current_job()` 包在 `suppress` 里。** RQ 上下文缺失（比如单测直接调函数、或者有人写脚本手动跑）不该让任务本身炸掉。这是「辅助设施不能成为失败原因」原则的又一次应用。

③ **脱离 RQ 上下文时函数是纯的。** 没有副作用，只返回判定结果。这让单测可以直接调，不需要起 Redis。测试见 `tests/unit/test_job_failure.py:67`。

④ **就地把 `retries_left` 清零。** 这是关键手法 —— RQ 没有提供「这次失败别重试」的官方钩子，但我们可以改它读取的那个属性。

⑤ **两个条件都要满足。** `retryable and retries_left > 0`：异常可重试**且**还有额度。少了后半句会出现「额度已经用尽，但 `format_failure_message` 里还写着『RQ 将自动重试』」—— 又是一次撒谎。测试见 `test_exhausted_retries_reports_no_retry`。

### 3.4 为什么改内存属性是可靠的

这一点必须自己去源码里确认，不能靠猜。docstring 里记了确认结果（`app/core/job_failure.py:92-102`）：

```
## 为什么改内存属性是可靠的（照 rq 2.9 源码确认过）

- `Job.perform()` 里 `_job_stack.push(self)`，所以 `get_current_job()`
  拿到的就是 worker 正在执行的**同一个 Job 对象**，不是副本。
- `Worker.perform_job()` 的 except 分支把这个对象直接传给
  `handle_job_failure(job=job, ...)`。
- `handle_job_failure` 里 `retry = job.should_retry and not job_is_stopped`，
  而 `should_retry` 读的是内存属性 `retries_left`，中间没有
  `job.refresh()`，不会从 Redis 重新取值。
```

实测确认过（rq 2.9.1）：

```
def should_retry(self) -> bool:
        return self.retries_left is not None and self.retries_left > 0
```

三个环节连起来才成立：同一个对象 → 直接传递 → 读内存不刷新。**任何一环变了这个手法就失效**，所以注释里写了 rq 版本号。这是依赖三方库内部行为时必须付的代价 —— 要么写清版本和依据，要么别这么写。

### 3.5 失败原因文案（`app/core/job_failure.py:131-143`）

```python
def format_failure_message(exc: BaseException, *, will_retry: bool) -> str:
    """拼装落库用的失败原因。

    带上 `error_code` 是为了可聚合 —— 自由文本没法做「同一类失败出现了多少次」
    的统计，而 `error_code` 是有限枚举。

    带上重试状态是为了**不误导看页面的人**：状态列写着 FAILED，但后台其实
    还排着两次重试。不说清楚的话，值班同学会立刻开始手工排查一个
    十秒后可能自己就好了的问题。
    """
    code = error_code_of(exc)
    suffix = "，RQ 将自动重试" if will_retry else ""
    return f"[{code}] {exc}{suffix}"
```

产出形如 `[parse_error] PDF 第 3 页字体描述损坏`，或 `[retrieval_error] milvus 连接超时，RQ 将自动重试`。

两个设计点：

**`error_code` 前缀让 error_message 可聚合。** 一句 `str(exc)` 是自由文本，每次都不一样（带着不同的文件名、不同的行号）。`SELECT count(*) FROM index_job WHERE error_message LIKE '[parse_error]%'` 能立刻回答「这周有多少份 PDF 是坏的」，自由文本做不到。

**重试状态是给人看的。** 这是整个降级改造的一贯主题：状态列写着 FAILED，是**事实**；但它不是**完整的事实**。缺的那半句会让值班同学做出错误动作。

### 3.6 落盘：绝不外抛（`app/core/job_failure.py:146-183`）

```python
def record_job_failure(
    write_status: Callable[[Session], None],
    *,
    job_id: str,
) -> None:
    """用独立 session 写入任务失败状态；二次失败只记日志，绝不外抛。

    这个函数的全部价值在于「绝不 raise」：抢救失败是坏消息，
    但让抢救过程中的异常顶掉原始异常，是更坏的消息。
    """
    try:
        session = SessionLocal()
    except Exception as exc:
        # 连建 session 都失败（连接池耗尽 / 数据库彻底不可达），
        # 那就只能认了 —— 至少把这件事记下来。
        logger.error(f"任务失败状态落盘失败（无法建立 session）: job_id={job_id}: {exc}")
        return                                        # ① 建不出来就放弃，不抛

    try:
        write_status(session)
        session.commit()
    except Exception as secondary:
        # 关键：这里绝不 raise。
        logger.error(
            f"任务失败状态落盘失败（原始异常已保留）: job_id={job_id}, "
            f"二次异常={type(secondary).__name__}: {secondary}"
        )                                             # ② 二次异常降级成日志
        with suppress(Exception):
            session.rollback()                        # ③ rollback 自己也可能抛
    finally:
        # close 在连接已断时同样可能抛。若让它从 finally 里逃出去，
        # 就又变成了「二次异常顶掉原始异常」—— 正是本模块要消灭的那个 bug。
        with suppress(Exception):
            session.close()                           # ④ finally 里也要 suppress
```

四个位置全都被封死了，这个函数**不存在任何抛出路径**。

注意 ④ 特别容易漏。很多人记得给 `commit` 包 try，但 `finally: session.close()` 看起来太无害了 —— 而它恰恰在最外层，抛出来一样会顶掉原始异常。测试专门覆盖了这条（`tests/unit/test_job_failure.py:127`）：

```python
    def test_close_failure_never_escapes(self) -> None:
        """close 在连接已断时同样会抛，从 finally 里逃出去就又变成顶掉原始异常。"""
        session = MagicMock()
        session.close.side_effect = RuntimeError("socket is dead")

        with patch("app.core.job_failure.SessionLocal", return_value=session):
            record_job_failure(lambda s: None, job_id="j4")
```

这个测试没有 `assert` —— **不抛异常就是通过**。这是测试「绝不抛出」这类契约的标准写法。

**为什么参数是回调而不是数据。** `record_job_failure` 不知道也不该知道 `IndexJob` 和 `ProtocolIngestionJob` 的字段长什么样。它只提供「一个干净的 session + 绝不外抛的执行环境」，具体写什么由调用方用闭包传进来。这是模板方法模式：**不变的部分（session 生命周期 + 异常封堵）在这里，变化的部分（写哪些字段）由调用方注入。**

### 3.7 worker 侧改造后的 except 块（`app/workers/index_worker.py:105-147`）

```python
    except Exception as exc:
        # 顺序是刻意的：先裁决重试，再落盘，最后原样 raise。
        #
        # 先裁决是因为 error_message 里要写清「还会不会重试」——
        # 否则页面上写着 FAILED、后台其实还排着两次重试，
        # 值班同学会立刻开始排查一个十秒后可能自己就好了的问题。
        will_retry = apply_retry_policy(exc, job_id=job_id)      # ①
        message = format_failure_message(exc, will_retry=will_retry)

        # 原 session 的连接可能已经废了，rollback 本身都可能抛。
        # 包进 suppress：它只是尽力释放事务，失败不该影响后面的抢救。
        with suppress(Exception):
            db.rollback()                                        # ②

        def _write_failed(session: Session) -> None:
            # 注意必须用新 session 重新构造 repo、重新取行。
            # 原 session 里的 ORM 对象绑在那个可能已经断掉的连接上，
            # 跨 session 复用会在 flush 时炸开。
            failure_repo = KnowledgeRepository(session)           # ③
            failed_job = failure_repo.get_index_job(job_id)
            if failed_job is None:
                return
            failed_job.status = IndexJobStatus.FAILED
            failed_job.error_message = message
            failed_job.finished_at = datetime.now(UTC).replace(tzinfo=None)
            failed_document = failure_repo.get_document(failed_job.document_id)
            if failed_document is not None:
                failed_document.status = DocumentStatus.FAILED
                failed_document.error_message = message

        # 走独立 session，且内部绝不外抛 —— 抢救失败最多丢一条状态，
        # 不能让二次异常顶掉真正的原因。
        record_job_failure(_write_failed, job_id=job_id)          # ④

        logger.opt(exception=exc).error(
            f"索引任务失败: job_id={job_id}, error_code={error_code_of(exc)}, "
            f"will_retry={will_retry}"                            # ⑤
        )
        # 必须原样 raise：RQ 靠这个异常判定任务失败并决定是否重排队。
        raise                                                     # ⑥
    finally:
        with suppress(Exception):
            db.close()                                            # ⑦
```

① **顺序不能换。** 裁决必须在落盘之前，因为 `message` 里要带重试状态。如果先落盘再裁决，写进去的文案就少了那半句。

② **`db.rollback()` 包 suppress。** 它的作用只是尽力释放原 session 上的事务（让连接能还给池子）。失败不影响后续 —— 因为后续用的是**新 session**，跟这个废掉的完全无关。

③ **闭包里重新构造 repo、重新取行。** 这是 SQLAlchemy 的硬约束：ORM 对象绑定在创建它的 session 上（`instance._sa_instance_state.session_id`）。把原 session 的 `job` 对象拿到新 session 里改字段，flush 时会抛 `InvalidRequestError: Object is already attached to session`。必须用新 session 重新 `get`。

④ **`record_job_failure` 是唯一的落盘出口。** 两个 worker 都走它，不各自实现一遍。

⑤ **`logger.opt(exception=exc)` 而不是 `logger.exception`。** 后者只能在 except 块里用且必须是当前异常；前者可以显式指定异常对象，语义更清楚。日志里带 `error_code` 和 `will_retry`，让日志本身就能回答「这次会不会自己恢复」。

⑥ **裸 `raise`，不是 `raise exc`。** 裸 `raise` 保留原始 traceback；`raise exc` 会把 traceback 的起点改成这一行，丢掉出错的真实位置。

⑦ **`finally` 里的 `close` 也要 suppress。** 同 §3.6 的 ④，理由一样。

### 3.8 入队侧给额度（`app/services/index_job_queue.py:11-28`）

```python
def enqueue_index_job(job_id: str) -> Job:
    # retry 必须在**入队时**声明：RQ 把重试额度存进 job 记录，worker 执行时
    # 只能减少它，不能凭空补上。也就是说没有这个参数的任务，一次失败就是终局，
    # 哪怕失败原因只是 Milvus 正在选主这种几秒钟就恢复的抖动。
    # 至于「哪些失败值得重试」，由 worker 侧的 apply_retry_policy 按异常类型裁决。
    return enqueue_with_request_id(
        index_queue,
        run_index_job,
        job_id,
        job_timeout="30m",
        failure_ttl=7 * 24 * 60 * 60,
        result_ttl=7 * 24 * 60 * 60,
        retry=default_job_retry(),          # ← 新增
    )
```

这里有个不对称，值得记住：

- **额度只能在入队时给。** RQ 把 `retries_left` 写进 Redis 里的 job hash。worker 执行时能读、能减，但不能无中生有 —— 因为 `handle_job_failure` 里的重排队逻辑是 `if retry:` 分支，`retry` 为假就走终局路径了。
- **裁决只能在执行时做。** 入队时根本不知道会失败，更不知道会因为什么失败。

所以这两个决策天然分居两处。它们不是重复，是**正交的两个维度**：入队侧回答「最多几次」，执行侧回答「这次值不值」。

---

## 4. 背后的工程原理

### 4.1 异常屏蔽（Exception Masking）

这是本篇的核心概念，几乎每种语言都有对应机制在处理它。

**问题定义：** 当异常 A 触发了清理代码，而清理代码抛出异常 B，如果 B 覆盖了 A，那么真正的故障原因就丢了。

几种语言的应对：

| 语言 | 机制 | 行为 |
|---|---|---|
| Java | try-with-resources | `close()` 抛的异常挂在原异常的 `getSuppressed()` 上，不覆盖 |
| Python | `raise ... from` / `__context__` | traceback 里保留链条，但**顶层异常仍然是 B** |
| Go | `defer` + 显式检查 | 语言不管，全靠程序员自己判断 |
| Rust | `Drop` 里 panic | double panic 直接 abort，语言强制你别这么干 |

Python 的处理最容易骗人：它**确实**保留了链条（`__context__`），traceback 里也会打出来。但顶层异常是 B —— 而所有只取顶层的地方（`str(exc)`、`type(exc).__name__`、RQ 的 `exc_info` 首行、告警标题、监控指标标签）看到的全是 B。

结论：**在清理路径上不要抛异常。** 不是「抛了也没事因为有 `__context__`」，而是真的不要抛。这是本篇所有 `with suppress(Exception)` 的唯一理由。

面试可以这样答：Python 的异常链保留了上下文，但顶层异常会被替换。生产环境里大部分消费方只读顶层，所以异常屏蔽在 Python 里同样是真实问题。工程做法是把清理路径的异常降级成日志。

### 4.2 为什么必须用独立 session

SQLAlchemy 的 Session 在异常后进入需要 rollback 的状态。这时对它做任何操作都会抛 `PendingRollbackError`。

但更根本的问题在下一层：**连接**。如果原始异常是数据库连接断开（服务端重启、网络分区、DBA kill 了长事务），那么这个 session 持有的连接已经是个死 socket。在它上面：

- `rollback()` → 抛（要发 ROLLBACK 命令过去）
- `commit()` → 抛
- 任何 `SELECT` → 抛
- `close()` → 可能抛

**在漏水的船上补漏。** 新建 session 会从连接池取一个新连接（如果池子里的连接也都废了，SQLAlchemy 的 pre-ping / 失效检测会重建），这才有机会写成功。

这条推广开来是一个更一般的原则：**故障恢复路径不能依赖已经故障的组件。** 同类的例子：

- 告警系统不能部署在被监控的集群里
- 日志上报失败时的报错不能再走日志系统
- 熔断器的状态不能存在被熔断的下游里

### 4.3 重试的前提是幂等

这一点在代码里没有注释，但它是整个重试设计能成立的隐含前提，也是面试高频题。

**如果任务不幂等，重试会造成数据重复或状态错乱。** 看 `app/workers/index_worker.py` 里的两个关键动作：

```python
58:        vector_store_manager.delete_by_document_id(document.id)
...
73:        repo.replace_chunks(document.id, chunks)
```

两个都是「先清后写」语义，不是「追加」：

- `delete_by_document_id` 先删掉这个文档在 Milvus 里的所有旧向量，再 `add_documents`
- `replace_chunks` 而不是 `add_chunks`

所以任务跑第二遍时，第一遍留下的半成品会被清掉。**重试三次和成功跑一次的最终状态一致。**

如果当初写的是 `add_documents` 不带删除，那么重试三次会在 Milvus 里留下三份重复向量 —— 检索时同一段内容占掉 top-k 的三个位置，召回质量直接崩。而这个 bug 只在「首次失败 + 重试成功」的组合下出现，正常路径测不出来。

**引入重试时必须先审计幂等性。** 顺序不能颠倒：先确认任务幂等，再加 retry。反过来做等于给自己埋雷。

### 4.4 状态机必须保证离开中间态

任何异步任务都有至少三个状态：`PENDING → RUNNING → {SUCCEEDED | FAILED}`。

**中间态是给人看进度的，不是可以停留的终点。** 一个任务停在 `RUNNING` 意味着：

- 前端永远转圈
- 按 `status = FAILED` 触发的告警不会响
- 按 `status = RUNNING AND started_at < now() - 1h` 的巡检（如果有的话）才能发现它

所以「保证写入终态」的优先级要高于「写入终态的内容是否完整」。`record_job_failure` 宁可写一条不完美的失败记录，也不能什么都不写。

顺带一句：即使有了这一层，生产系统仍然应该配一个**兜底巡检** —— 扫描 `RUNNING` 且 `started_at` 超过 `job_timeout` 两倍的行，标成 FAILED。因为总有 worker 进程被 OOM kill、被 `SIGKILL` 的情况，那时候 `except` 块根本没机会执行。这属于 YAGNI 边界：当前没配，但知道它是缺口。

### 4.5 正交决策要放在各自的正确位置

重试这件事被拆成了两个决策：

```
入队时（index_job_queue.py）        执行时（job_failure.py）
    ↓                                  ↓
「最多重试几次」                    「这次值不值得重试」
retry=default_job_retry()          apply_retry_policy(exc)
    ↓                                  ↓
写进 Redis job hash                改内存 retries_left
```

判断「这段逻辑该放哪」的方法是问：**它依赖的信息在哪里可得？**

- 「最多几次」依赖的是业务对这类任务的容忍度 —— 入队时就知道，执行时也不会变
- 「这次值不值」依赖的是**具体这次抛的异常** —— 只有执行时才存在

信息在哪里，决策就在哪里。硬把两者塞到一处，必然要么传参数传一堆，要么在错误的地方猜。

### 4.6 分类逻辑的唯一实现处（DRY）

`is_retryable` 的调用方有三个：worker 的重试裁决、熔断器的计数判定、HTTP 层的状态码映射。

如果各自 `isinstance` 一遍，会出现这种情况：worker 认为 `RateLimitError` 可重试（对），熔断器认为它不算故障（也对），而 HTTP 层认为它是 400（错了，应该 429/502）。三处不一致，而且改一处不会同步另外两处。

集中到 `errors.py:222-277` 之后，新增一种异常类型只需要往 frozenset 里加一个名字，三个消费方同时正确。

注意实现细节（`errors.py:271`）：

```python
    for klass in type(exc).__mro__:
```

走 MRO 而不是只看 `type(exc).__name__`。因为 openai SDK 的 `APITimeoutError` 继承自 `APIConnectionError`，只看自身类名会漏掉继承关系带来的语义。用**类名而不是类对象**匹配则是为了避免 `import openai` —— 三方包不该成为 `errors.py` 的硬依赖。

---

## 5. 怎么验证

### 5.1 跑失败处置的单元测试

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -m pytest tests/unit/test_job_failure.py -p no:cacheprovider -q'
```

期望：12 passed。

### 5.2 确认「绝不外抛」这条契约

最关键的三个用例单独跑：

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  .venv/bin/python -m pytest tests/unit/test_job_failure.py::TestRecordJobFailure \
  -p no:cacheprovider -v'
```

看这三条：
- `test_commit_failure_never_masks_original_exception`
- `test_session_construction_failure_is_swallowed`
- `test_close_failure_never_escapes`

它们都没有 `assert` —— **函数不抛异常就是通过**。

### 5.3 亲手验证异常屏蔽

这段可以直接贴进 python 跑，看清「二次异常顶掉原始异常」到底长什么样：

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && .venv/bin/python - <<"PY"
# 旧写法：except 里再抛
def old_style():
    try:
        raise ValueError("PDF 第 3 页字体描述损坏")   # 真正的原因
    except Exception:
        raise RuntimeError("server closed the connection")  # 抢救时炸了

try:
    old_style()
except Exception as e:
    print("顶层异常     :", type(e).__name__, "-", e)
    print("__context__ :", type(e.__context__).__name__, "-", e.__context__)
    print()
    print("→ 如果监控只取 str(exc)，看到的是:", str(e))
    print("→ 真正要修的东西藏在 __context__ 里")
PY' 2>&1 | tr -d "\r"
```

期望输出：顶层是 `RuntimeError`，真正的 `ValueError` 沉在 `__context__`。

### 5.4 确认重试裁决真的会改 job

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && .venv/bin/python - <<"PY"
from unittest.mock import MagicMock, patch
from app.core.errors import ParseError, RetrievalError
from app.core.job_failure import apply_retry_policy, format_failure_message

def probe(exc, retries_left):
    job = MagicMock()
    job.retries_left = retries_left
    with patch("app.core.job_failure.get_current_job", return_value=job):
        will_retry = apply_retry_policy(exc, job_id="probe")
    return will_retry, job.retries_left

for exc, left in [
    (RetrievalError("milvus 抖动"), 3),
    (ParseError("PDF 损坏"), 3),
    (RetrievalError("还是抖"), 0),
]:
    will_retry, after = probe(exc, left)
    msg = format_failure_message(exc, will_retry=will_retry)
    print(f"{type(exc).__name__:16} 额度 {left} → will_retry={will_retry!s:5} 剩余={after}")
    print(f"                 落库文案: {msg}")
PY' 2>&1 | tr -d "\r"
```

期望：可重试的保留额度 3；不可重试的被清成 0；额度已尽的 `will_retry=False` 且文案里没有「将自动重试」。

### 5.5 确认两个 worker 都走了统一出口

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  grep -rn "record_job_failure\|apply_retry_policy" app/workers/'
```

期望：两个 worker 各出现两次（import + 调用），没有任何 worker 自己写落盘逻辑。

### 5.6 确认旧的「同 session 抢救」写法已清零

```bash
wsl.exe -d Ubuntu -- bash -lc 'cd /home/dong/projects/super_biz_agent_py && \
  grep -n "db.commit()" app/workers/*.py'
```

期望：只在正常路径（`try` 块里）出现，`except` 块里一个都没有。

---

## 6. 常见坑

### 6.1 MagicMock 的属性默认是 MagicMock，比较运算恒真

`tests/unit/test_job_failure.py:28` 专门写了个辅助函数，注释里点了这件事：

```python
def _fake_job(retries_left: int) -> MagicMock:
    """造一个只带 retries_left 的假 job。

    必须显式赋整数：MagicMock 的属性默认是 MagicMock，
    而 `retries_left > 0` 对 MagicMock 恒为真值，会让断言失去意义。
    """
    job = MagicMock()
    job.retries_left = retries_left
    return job
```

如果忘了赋值，`job.retries_left` 是个 MagicMock，`MagicMock() > 0` 返回的是**另一个 MagicMock**，而它是真值。于是 `retries_left > 0` 恒成立，测试「通过」了但什么都没验证到。

这是 mock 测试里最常见的假绿。规律是：**凡是要参与比较、算术、布尔判断的属性，都必须显式赋真实类型的值。**

### 6.2 `retries_left = 0` 还是 `= None`

`should_retry` 的实现是 `self.retries_left is not None and self.retries_left > 0`，所以两种写法都能让 RQ 不重试。选 `0` 的理由是语义：

- `None` = 「这个任务没配过重试」
- `0` = 「配过，但额度用完了 / 被否决了」

排障时看到 `retries_left=0` 加上那条 `已取消剩余 3 次重试` 的 warning，能还原出完整故事。看到 `None` 会以为入队时忘了加 retry 参数。

### 6.3 千万不要在 except 块里 `return`

如果 worker 的 except 块最后写的是 `return` 而不是 `raise`：

```python
    except Exception as exc:
        record_job_failure(_write_failed, job_id=job_id)
        return          # ← 灾难
```

RQ 会认为这个任务**成功了**。后果是：

- job 进 finished registry，不进 failed registry
- 重试完全不会发生（没有异常，`handle_job_failure` 都不会被调用）
- 数据库里状态是 FAILED，Redis 里状态是 finished —— 两个数据源互相矛盾

数据源之间的矛盾比单纯的失败难查得多，因为你会怀疑是自己看错了。

### 6.4 重试后 request_id 还在吗

在。这是个容易担心但实际没问题的点：

`request_id` 存在 `job.meta` 里（见 [第 15 篇](15-request_id全链路追踪.md)），而 RQ 重排队时是把**同一个 job** 重新入队，`meta` 跟着走。所以四次尝试的日志共享同一个 rid，`grep <rid>` 能看到完整的四次尝试。

反倒要注意的是：同一个 rid 出现四次会让日志看起来像是同一次请求被处理了四遍。日志里的 `will_retry=True` 就是用来区分这个的。

### 6.5 `job_timeout` 和重试间隔是两码事

`job_timeout="30m"` 管的是**单次执行**的上限，不是四次尝试的总和。所以最坏情况是：

```
30m + 10s + 30m + 30s + 30m + 60s + 30m ≈ 2 小时
```

配的时候要意识到这个乘数关系。如果业务上「两小时后才知道失败」不可接受，要么缩 `job_timeout`，要么减重试次数 —— 不能只看单个参数。

### 6.6 `finally: db.close()` 不包 suppress

最容易漏的一个。人的直觉是「close 而已，能出什么事」，但连接已断时它真的会抛，而且它在最外层，抛出来就顶掉原始异常 —— 绕了一圈又回到了本篇要消灭的那个 bug。

`app/workers/index_worker.py:145-147` 和 `protocol_pdf_worker.py:91-93` 都是：

```python
    finally:
        with suppress(Exception):
            db.close()
```

### 6.7 加重试之前没审幂等

见 §4.3。这一条的危险在于**它不会立刻爆**：加了 retry 之后系统看起来更健壮了，直到某天检索质量莫名下降，才发现 Milvus 里有一堆重复向量，来自几个月前某次「重试成功」的任务。

顺序必须是：先确认幂等 → 再加重试。

### 6.8 别指望 except 块一定会执行

worker 进程被 OOM killer 干掉、被 `kill -9`、机器断电 —— 这些情况下 `except` 块根本没机会跑。所以 §4.4 提到的兜底巡检是必要的，只靠 `except` 里的落盘保证不了「一定离开中间态」。

当前项目没做这个巡检（YAGNI，单机部署且任务量不大），但这是一个**已知缺口**，不是没想到。面试被问「你的异步任务如何保证不卡在中间态」时，答案应该包含两层：进程内的 except 落盘 + 进程外的超时巡检。

---

## 7. 一句话总结

> 清理路径上的异常会顶掉真正的故障原因，所以抢救必须用独立 session 且绝不外抛；
> 重试拆成「入队给额度」和「执行时按异常类型裁决」两个正交决策；
> 而重试能成立的前提是任务幂等 —— 这一条要在加 retry 之前先审。
