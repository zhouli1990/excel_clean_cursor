import logging
import logging.handlers
import sys
import os
import glob
import time
from datetime import datetime, timedelta
from typing import Optional, List

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
FULL_CONTEXT_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - "
    "%(funcName)s - %(process)d - %(thread)d - %(message)s"
)

# 日志目录
LOG_DIR = "logs"
# 最大日志文件大小 (5MB)
MAX_LOG_SIZE = 30 * 1024 * 1024
# 最大日志备份数量
MAX_LOG_BACKUP_COUNT = 20
# 日志保留天数
LOG_RETENTION_DAYS = 7


def setup_logger(
    name: str,
    level: str = "debug",
    console: bool = True,
    file: bool = True,
    detailed_format: bool = False,
    full_context: bool = False,
    max_bytes: int = MAX_LOG_SIZE,
    backup_count: int = MAX_LOG_BACKUP_COUNT,
) -> logging.Logger:
    """
    配置并返回一个日志记录器。

    Args:
        name: 日志记录器名称
        level: 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
        console: 是否输出到控制台
        file: 是否输出到文件
        detailed_format: 是否使用详细格式
        full_context: 是否包含完整上下文（函数名、进程ID、线程ID等）
        max_bytes: 单个日志文件最大大小
        backup_count: 日志文件最大备份数量

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取日志级别
    log_level = LOG_LEVELS.get(level.lower(), logging.DEBUG)

    # 选择格式化器
    if full_context:
        log_format = FULL_CONTEXT_FORMAT
    elif detailed_format:
        log_format = DETAILED_FORMAT
    else:
        log_format = DEFAULT_FORMAT

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

        # 添加轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def clean_old_logs(days: int = LOG_RETENTION_DAYS) -> int:
    """
    清理旧的日志文件。

    Args:
        days: 保留的天数（默认7天）

    Returns:
        int: 已删除的文件数量
    """
    # 确保日志目录存在
    if not os.path.exists(LOG_DIR):
        return 0

    # 计算截止日期
    cutoff_date = datetime.now() - timedelta(days=days)
    deleted_count = 0

    try:
        # 获取所有日志文件
        log_files = glob.glob(f"{LOG_DIR}/*.log*")

        for log_file in log_files:
            try:
                # 获取文件修改时间
                file_mtime = datetime.fromtimestamp(os.path.getmtime(log_file))

                # 如果文件早于截止日期，则删除
                if file_mtime < cutoff_date:
                    os.remove(log_file)
                    deleted_count += 1
            except Exception as e:
                print(f"清理日志文件 {log_file} 失败: {e}")

        return deleted_count
    except Exception as e:
        print(f"清理日志文件过程中出错: {e}")
        return deleted_count


def get_log_files(name: Optional[str] = None, days: int = 7) -> List[str]:
    """
    获取指定名称和时间范围内的日志文件列表。

    Args:
        name: 日志名称过滤（None表示所有）
        days: 最近的天数范围

    Returns:
        List[str]: 日志文件路径列表
    """
    # 确保日志目录存在
    if not os.path.exists(LOG_DIR):
        return []

    # 构建搜索模式
    if name:
        pattern = f"{LOG_DIR}/{name}_*.log*"
    else:
        pattern = f"{LOG_DIR}/*.log*"

    # 获取所有匹配的日志文件
    log_files = glob.glob(pattern)

    # 如果指定了天数，筛选出最近N天内的文件
    if days > 0:
        cutoff_date = datetime.now() - timedelta(days=days)
        log_files = [
            f
            for f in log_files
            if datetime.fromtimestamp(os.path.getmtime(f)) >= cutoff_date
        ]

    # 按修改时间排序（最新的在前）
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    return log_files


# 启动时自动清理旧日志
clean_old_logs()
