import os
import logging
import ray
import sys


def configure_logging(suppress_all=False):
    """配置日志系统，禁用所有不必要的输出"""

    # 禁用Flower所有INFO及以下日志
    os.environ["FLWR_LOG_LEVEL"] = "error"  # 只显示ERROR及以上级别
    os.environ["FLWR_SERVER_LOG_LEVEL"] = "error"
    os.environ["FLWR_CLIENT_LOG_LEVEL"] = "error"

    # 禁用Ray所有INFO及以下日志
    os.environ["RAY_LOG_LEVEL"] = "error"  # 只显示ERROR及以上级别
    os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"
    os.environ["RAY_LOG_TO_STDERR"] = "0"
    os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"  # 忽略未处理的错误

    # 配置Python标准日志系统
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("flwr").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.actor").setLevel(logging.ERROR)
    logging.getLogger("ray.worker").setLevel(logging.ERROR)

    # 自定义过滤器，屏蔽包含"ClientAppActor"的日志
    class ClientAppActorFilter(logging.Filter):
        def filter(self, record):
            return "ClientAppActor" not in record.getMessage()

    # 应用过滤器到所有日志处理程序
    for handler in logging.root.handlers:
        handler.addFilter(ClientAppActorFilter())

    # 初始化Ray（确保在设置环境变量后）
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.ERROR,
            logging_format="[%(asctime)s] [%(levelname)s] %(message)s",
            include_dashboard=False,
            ignore_reinit_error=True,
            num_cpus=1,  # 限制CPU使用，减少进程数量
        )

    # 可选：完全抑制标准输出和标准错误（极端措施）
    if suppress_all:
        class DummyFile:
            def write(self, x): pass

            def flush(self): pass

        sys.stdout = DummyFile()
        sys.stderr = DummyFile()

    print("日志系统已配置：仅显示ERROR级别的信息")