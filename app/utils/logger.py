"""日志配置模块

使用 Loguru 配置应用日志。

两个关键设计：

1. **request_id 自动注入**（配合 app/core/request_context.py）
   通过 `logger.configure(patch=...)` 给每条日志记录补上 `extra["rid"]`，
   格式串里再用 `{extra[rid]}` 输出。这样业务代码一行都不用改，
   所有既有的 `logger.info(...)` 自动获得 request_id。

   为什么必须用 patch 而不是让每个调用点自己 bind：
   项目里有数百处 logger 调用，还有 langchain / uvicorn 等第三方库的日志。
   patch 是唯一的「一处实现，全局生效」的切面（DRY）。

2. **文件输出 JSON 化**
   控制台保留彩色文本给人看，文件用 `serialize=True` 输出 JSON 给机器看。
   纯文本日志要接 Loki/ELK 必须先写正则解析，字段一变解析就崩；
   JSON 行天然结构化，`rid` / `level` / `module` 直接就是可查询字段。
"""

import sys

from loguru import logger

from app.config import config
from app.core.request_context import NO_REQUEST_ID, get_request_id


def _patch_request_id(record: dict) -> None:
    """给每条日志补上 rid 字段。

    用 setdefault 而不是直接赋值，有两个原因：
    1. 保留调用方的显式意图 —— `logger.bind(rid="job-123")` 不该被覆盖。
    2. 防 KeyError —— 格式串里写了 `{extra[rid]}`，一旦某条记录缺这个键，
       loguru 格式化时会抛 KeyError，而**日志系统自身抛异常是最糟的故障**：
       它会掩盖真正的错误。setdefault 保证这个键永远存在。
    """
    record["extra"].setdefault("rid", get_request_id() or NO_REQUEST_ID)


def setup_logger():
    """配置日志系统

    按照 Loguru 最佳实践配置全局 logger：
    1. 移除默认处理器
    2. 注册 patcher（注入 request_id）
    3. 添加控制台输出（带颜色，给人读）
    4. 添加文件输出（JSON 行，按天轮转，给机器查）
    """
    # 移除默认处理器
    logger.remove()

    # 注册 patcher：必须在 add() 之前，否则已注册的 handler 拿不到 rid。
    # 注意关键字是 patcher（不是 patch）—— loguru 里 `logger.patch()` 是返回
    # 新 logger 的方法，而 `configure(patcher=...)` 才是给全局装切面。
    logger.configure(patcher=_patch_request_id)

    # 添加控制台输出（带颜色格式）
    # rid 放在 level 之后、模块之前：一眼能看到「这条属于哪个请求」
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>{extra[rid]}</magenta> | "
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,  # 显示完整异常栈信息
        diagnose=config.debug,  # Debug 模式下显示变量值
    )

    # 添加文件输出（JSON 行，按天轮转，自动压缩）
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # 每天0点自动切割新日志文件
        # 保留 30 天：故障 case 常常几周后才被复盘追问，7 天太短，
        # 等真要查的时候日志已经被删了。压缩后单日体积很小，30 天完全可接受。
        retention="30 days",
        compression="zip",  # 过期日志自动压缩为zip
        encoding="utf-8",  # 解决中文乱码
        enqueue=True,  # 异步写入，提升性能（避免IO阻塞）
        backtrace=True,  # 显示完整异常栈信息
        diagnose=True,  # 显示变量值，便于调试
        level="INFO",
        # serialize=True 输出 JSON 行，record.extra.rid 可直接被日志系统索引。
        # 注意：serialize 时 format 不参与最终输出（loguru 输出完整 record 的 JSON），
        # 但仍保留 format 以便 message 字段渲染一致。
        serialize=True,
        format="{message}",
    )


setup_logger()
