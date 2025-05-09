import logging
import sys
import os
from datetime import datetime

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# 日志格式化器
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# 日志目录
LOG_DIR = "logs"


def setup_logger(
    name: str,
    level: str = "info",
    console: bool = True,
    file: bool = True,
    detailed_format: bool = False,
):
    """
    配置并返回一个日志记录器。

    Args:
        name: 日志记录器名称
        level: 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
        console: 是否输出到控制台
        file: 是否输出到文件
        detailed_format: 是否使用详细格式

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取日志级别
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)

    # 选择格式化器
    log_format = DETAILED_FORMAT if detailed_format else DEFAULT_FORMAT
    formatter = logging.Formatter(log_format)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 防止重复添加处理器
    if logger.handlers:
        logger.handlers.clear()

    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 添加文件处理器
    if file:
        # 确保日志目录存在
        os.makedirs(LOG_DIR, exist_ok=True)

        # 创建日志文件名（按日期）
        date_str = datetime.now().strftime("%Y%m%d")
        log_filename = f"{LOG_DIR}/{name}_{date_str}.log"

        # 添加文件处理器
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
