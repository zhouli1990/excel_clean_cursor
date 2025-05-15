import os
import pandas as pd
import tempfile
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# 导入日志工具
from utils.logger import setup_logger
from utils.error_handler import try_except_with_logging

# 设置日志记录器
logger = setup_logger("file_utils")

# 定义文件大小警告阈值 (50MB)
FILE_SIZE_WARNING_THRESHOLD = 50 * 1024 * 1024

# 定义行数警告阈值 (10000行)
ROW_COUNT_WARNING_THRESHOLD = 10000


@try_except_with_logging(default_value=None, error_message="创建目录失败")
def ensure_dir_exists(directory: str) -> bool:
    """
    确保目录存在，不存在则创建

    Args:
        directory: 目录路径

    Returns:
        bool: 是否成功创建或已存在
    """
    try:
        if os.path.exists(directory):
            if not os.path.isdir(directory):
                logger.warning(f"路径存在但不是目录: {directory}")
                return False
            logger.debug(f"目录已存在: {directory}")
            return True

        os.makedirs(directory, exist_ok=True)
        logger.info(f"已创建目录: {directory}")
        return True
    except Exception as e:
        logger.error(f"创建目录 {directory} 失败: {e}")
        return False


@try_except_with_logging(default_value=None, error_message="读取文件失败")
def read_excel_or_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    读取Excel或CSV文件

    Args:
        file_path: 文件路径

    Returns:
        Optional[pd.DataFrame]: 读取的数据，失败则返回None
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None

    # 检查文件大小
    file_size = os.path.getsize(file_path)
    formatted_size = format_file_size(file_size)

    if file_size > FILE_SIZE_WARNING_THRESHOLD:
        logger.warning(f"读取的文件超过警告阈值({formatted_size}): {file_path}")
    else:
        logger.info(f"开始读取文件({formatted_size}): {file_path}")

    start_time = time.time()
    try:
        if file_path.endswith((".xlsx", ".xls")):
            # 添加converters参数确保电话列为字符串
            # 使用可能的电话列名作为字符串转换
            possible_phone_cols = [
                "电话",
                "手机",
                "电话号码",
                "手机号码",
                "联系方式",
                "电话/手机",
            ]
            converters = {col: str for col in possible_phone_cols}

            logger.debug(f"使用excel引擎读取文件: {file_path}")
            df = pd.read_excel(file_path, engine="openpyxl", converters=converters)
            logger.debug(f"Excel引擎读取完成，正在处理数据")

        elif file_path.endswith(".csv"):
            try:
                logger.debug(f"尝试使用UTF-8编码读取CSV: {file_path}")
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8解码失败，尝试使用GBK编码: {file_path}")
                df = pd.read_csv(file_path, encoding="gbk")
                logger.debug(f"成功使用GBK编码读取CSV")
        else:
            err_msg = f"不支持的文件格式: {file_path}, 请使用Excel(.xlsx, .xls)或CSV(.csv)文件"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # 记录读取时间
        elapsed_time = time.time() - start_time

        # 清理列名（去除空格）
        df.columns = df.columns.str.strip()

        # 添加local_row_id
        if "local_row_id" not in df.columns:
            df["local_row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
            logger.info(f"已为数据添加'local_row_id'列")
        else:
            # 处理已存在但可能有空值或重复的情况
            existing_ids = df["local_row_id"].dropna().astype(str)
            if len(existing_ids) != len(df) or existing_ids.duplicated().any():
                logger.warning(f"'local_row_id'列存在但有空值或重复，重新生成ID")
                df["local_row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # 记录行列统计和处理时间
        row_count = len(df)
        col_count = len(df.columns)

        # 针对大量数据发出警告
        if row_count > ROW_COUNT_WARNING_THRESHOLD:
            logger.warning(f"数据量较大 ({row_count} 行), 处理可能较慢")

        logger.info(
            f"成功读取 {row_count} 行 {col_count} 列数据，耗时 {elapsed_time:.2f} 秒: {file_path}"
        )
        return df

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"读取文件失败 {file_path}: {e}, 耗时 {elapsed_time:.2f} 秒")
        return None


@try_except_with_logging(default_value=False, error_message="保存文件失败")
def save_dataframe(
    df: pd.DataFrame, file_path: str, sheet_name: str = "Sheet1", index: bool = False
) -> bool:
    """
    保存DataFrame到Excel或CSV文件

    Args:
        df: 要保存的DataFrame
        file_path: 保存路径
        sheet_name: Excel的sheet名称
        index: 是否保存行索引

    Returns:
        bool: 是否成功保存
    """
    try:
        # 确保目录存在
        output_dir = os.path.dirname(file_path)
        ensure_dir_exists(output_dir)

        # 记录开始保存
        row_count = len(df)
        col_count = len(df.columns)
        logger.info(f"开始保存 {row_count} 行 {col_count} 列数据到 {file_path}")

        start_time = time.time()

        # 根据文件扩展名决定保存方式
        if file_path.endswith((".xlsx", ".xls")):
            logger.debug(f"使用Excel格式保存数据到 {file_path}")
            df.to_excel(file_path, sheet_name=sheet_name, index=index)
        elif file_path.endswith(".csv"):
            logger.debug(f"使用CSV格式保存数据到 {file_path}")
            df.to_csv(file_path, index=index, encoding="utf-8")
        else:
            # 默认保存为Excel
            if not file_path.endswith((".xlsx", ".xls", ".csv")):
                file_path += ".xlsx"
                logger.debug(
                    f"未指定有效的文件扩展名，默认使用Excel格式保存到 {file_path}"
                )
            df.to_excel(file_path, sheet_name=sheet_name, index=index)

        # 记录保存耗时和文件大小
        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(file_path)
        formatted_size = format_file_size(file_size)

        logger.info(
            f"已成功保存 {row_count} 行数据到 {file_path}，文件大小 {formatted_size}，耗时 {elapsed_time:.2f} 秒"
        )
        return True

    except Exception as e:
        logger.error(f"保存数据到 {file_path} 失败: {e}")
        return False


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    获取文件信息

    Args:
        file_path: 文件路径

    Returns:
        Dict: 文件信息包括大小、修改时间等
    """
    logger.debug(f"获取文件信息: {file_path}")
    try:
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        modified_time = datetime.fromtimestamp(file_stat.st_mtime)

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)

        file_info = {
            "path": file_path,
            "name": os.path.basename(file_path),
            "size": file_size,
            "size_formatted": format_file_size(file_size),
            "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": ext.lstrip(".").upper() if ext else "UNKNOWN",
            "exists": True,
        }

        logger.debug(
            f"文件信息获取成功: {os.path.basename(file_path)}, 大小: {file_info['size_formatted']}"
        )
        return file_info
    except FileNotFoundError:
        logger.warning(f"文件不存在: {file_path}")
        return {
            "path": file_path,
            "name": os.path.basename(file_path),
            "exists": False,
            "error": "文件不存在",
        }
    except Exception as e:
        logger.error(f"获取文件信息失败 {file_path}: {e}")
        return {
            "path": file_path,
            "name": os.path.basename(file_path),
            "exists": True,
            "error": str(e),
        }


def format_file_size(size_in_bytes: int) -> str:
    """
    将文件大小格式化为人类可读的形式

    Args:
        size_in_bytes: 文件大小（字节）

    Returns:
        str: 格式化后的文件大小
    """
    # 定义单位
    units = ["B", "KB", "MB", "GB", "TB"]

    # 处理0字节情况
    if size_in_bytes == 0:
        return "0 B"

    # 计算单位
    i = 0
    while size_in_bytes >= 1024 and i < len(units) - 1:
        size_in_bytes /= 1024.0
        i += 1

    # 格式化输出 (保留两位小数)
    return f"{size_in_bytes:.2f} {units[i]}"
