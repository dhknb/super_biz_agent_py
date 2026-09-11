# ADR-006：测试隔离与密钥卫生

- 状态：已接受
- 日期：2026-07-23
- 关联：[[000]]、[[005]]

## 背景

测试套件在收集阶段会打印真实 DashScope API Key 的掩码
（`API Key: sk-ec7f1...7ebe`）到日志。虽然是掩码，但暴露了 key 的
头尾，且证明**真实 key 进入了测试进程**——测试本应完全离线、
不接触任何真实凭据。

根因有三层：

1. `vector_embedding_service.py`、`vector_store_manager.py` 是
   **模块级单例**，在 `import` 那一刻就构造并打日志。
2. `config = Settings()` 在 `import app.config` 时从 `.env` 读到真 key。
3. 原 conftest 用 `os.environ.setdefault(...)` 且放在 **autouse fixture**
   里——时机太晚（单例在收集期 import 时早已构造完），且 `.env` 的值
   不进 `os.environ`，`setdefault` 形同虚设。

## 决策

在 `tests/conftest.py` 的**模块顶层**（pytest 最先导入 conftest）、
任何 `app.*` import 之前，用**赋值**（而非 setdefault）强制设置假环境变量：

```python
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-test-fake-key-for-testing-only"
os.environ["MILVUS_HOST"] = "localhost"
os.environ["MILVUS_PORT"] = "19530"
# 之后才 import 其它模块
```

pydantic Settings 的优先级是「真实环境变量 > .env 文件」，所以顶层赋值
能让 `config` 拿到假 key；模块级单例即使在 import 期构造，打印的也是
`sk-test-...only` 假掩码，真 key 绝不进测试进程。

## 备选方案

- **给单例加惰性构造**（首次使用才连接）：更彻底，但要改动多个生产模块，
  风险大、超出两周范围，留作后续重构。
- **在 CI 里用假 .env**：治标不治本，本地跑测试仍会泄漏。
- **mock 掉 logger**：只是藏住症状，真 key 仍在进程里。

选顶层赋值是**成本最低、即刻见效**的治本手段。

## 后果

正面：
- 测试日志不再出现真实 key，实现凭据卫生。
- 测试与 `.env` 彻底解耦，离线可跑、可复现。
- 顺带清理了 3 个文件共 10 处废弃的 `datetime.utcnow()`
  （换成 `datetime.now(UTC).replace(tzinfo=None)`，与既有模型一致）。

负面 / 代价：
- conftest 顶层出现「import 前先赋值」的非常规写法，需要注释说明为什么
  不能挪到 import 之后（否则 E402 lint 也会告警）。

## 踩坑记录（工程纪律）

修这个 bug 时，Edit 工具在 WSL 网络路径（`//wsl.localhost/...`）上多次
**报告"成功"但实际没落盘**，导致「改了→测了→没变化」的假象重复出现。
最终靠 `cat -A` 直读磁盘字节 + 在 WSL 内用 Python 脚本改文件才根治。

纪律：**凡关键修改，一律用 `cat -A` / `sed -n` 直读磁盘验证，
不轻信工具的"成功"提示，尤其在网络文件系统上。**
