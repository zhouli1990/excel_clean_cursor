import os
import pandas as pd
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# 导入日志工具
from utils.logger import setup_logger
from utils.error_handler import try_except_with_logging

# 设置日志记录器
logger = setup_logger("file_utils")


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
        os.makedirs(directory, exist_ok=True)
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
            df = pd.read_excel(file_path, engine="openpyxl", converters=converters)
        elif file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8解码失败，尝试使用GBK: {file_path}")
                df = pd.read_csv(file_path, encoding="gbk")
        else:
            raise ValueError(
                "不支持的文件格式，请使用Excel(.xlsx, .xls)或CSV(.csv)文件"
            )

        # 清理列名（去除空格）
        df.columns = df.columns.str.strip()

        # 添加local_row_id
        if "local_row_id" not in df.columns:
            df["local_row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
            logger.info(f"已为数据添加'local_row_id'列: {file_path}")
        else:
            # 处理已存在但可能有空值或重复的情况
            existing_ids = df["local_row_id"].dropna().astype(str)
            if len(existing_ids) != len(df) or existing_ids.duplicated().any():
                logger.warning(
                    f"'local_row_id'列存在但有空值或重复，重新生成ID: {file_path}"
                )
                df["local_row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        logger.info(f"成功读取 {len(df)} 行数据: {file_path}")
        return df

    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
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

        # 根据文件扩展名决定保存方式
        if file_path.endswith((".xlsx", ".xls")):
            df.to_excel(file_path, sheet_name=sheet_name, index=index)
        elif file_path.endswith(".csv"):
            df.to_csv(file_path, index=index, encoding="utf-8")
        else:
            # 默认保存为Excel
            if not file_path.endswith((".xlsx", ".xls", ".csv")):
                file_path += ".xlsx"
            df.to_excel(file_path, sheet_name=sheet_name, index=index)

        logger.info(f"已成功保存 {len(df)} 行数据到 {file_path}")
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
    try:
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        modified_time = datetime.fromtimestamp(file_stat.st_mtime)

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)

        return {
            "path": file_path,
            "name": os.path.basename(file_path),
            "size": file_size,
            "size_formatted": format_file_size(file_size),
            "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": ext.lstrip(".").upper() if ext else "UNKNOWN",
            "exists": True,
        }
    except FileNotFoundError:
        return {
            "path": file_path,
            "name": os.path.basename(file_path),
            "exists": False,
            "error": "文件不存在",
        }
    except Exception as e:
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


def create_multi_sheet_excel(
    data_dict: Dict[str, pd.DataFrame], output_path: str, task_id: Optional[str] = None
) -> bool:
    """
    创建多Sheet页的Excel文件

    Args:
        data_dict: sheet名称到DataFrame的映射
        output_path: 输出文件路径
        task_id: 可选的任务ID (用于日志记录)

    Returns:
        bool: 是否成功创建
    """
    try:
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        ensure_dir_exists(output_dir)

        # 创建Excel Writer
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # 遍历数据字典，写入每个Sheet
            for sheet_name, df in data_dict.items():
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"已写入Sheet '{sheet_name}'，共 {len(df)} 行数据")
                else:
                    # 创建空DataFrame避免错误
                    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.warning(f"Sheet '{sheet_name}' 数据为空")

        task_info = f"[任务 {task_id}] " if task_id else ""
        logger.info(f"{task_info}已成功创建多Sheet页Excel文件: {output_path}")
        return True

    except Exception as e:
        task_info = f"[任务 {task_id}] " if task_id else ""
        logger.error(f"{task_info}创建多Sheet页Excel文件失败: {e}")
        return False
