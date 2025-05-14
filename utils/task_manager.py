import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# 从logger模块导入日志工具
from utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger("task_manager")

# 最大历史记录条目数
MAX_HISTORY_ENTRIES = 10
# 历史记录文件路径
HISTORY_FILE_PATH = "task_history.json"

# 全局变量 - 添加最近完成预处理的任务ID追踪
LATEST_COMPLETED_PREPROCESSING_TASK_ID = None

# 全局历史记录变量
TASK_HISTORY = []

# 任务状态字典
tasks = {}


def get_task_info(task_id: str) -> Optional[Dict]:
    """
    获取任务信息。

    Args:
        task_id: 任务ID

    Returns:
        Optional[Dict]: 任务信息字典，如果任务不存在则返回None
    """
    return tasks.get(task_id)


def create_task(files: List[str], task_type: str = "complete_processing") -> str:
    """
    创建新任务。

    Args:
        files: 任务关联的文件列表
        task_type: 任务类型，可以是'complete_processing'或'direct_import'

    Returns:
        str: 生成的任务ID
    """
    # 生成任务ID，基于时间戳和唯一ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_id = f"task_{timestamp}_{str(uuid.uuid4())[:8]}"

    # 初始化任务状态信息
    tasks[task_id] = {
        "status": "Queued",  # 状态：已入队
        "progress": 0,  # 进度百分比
        "total_files": len(files),  # 总文件数
        "files_processed": 0,  # 已处理文件数
        "result_file": None,  # 结果文件名（处理完成后设置）
        "error": None,  # 错误信息（如果发生错误）
        "task_type": task_type,  # 任务类型
    }

    # 添加到历史记录
    add_task_history_entry(task_id, task_type)

    logger.info(f"已创建新任务 {task_id}，类型为 {task_type}，关联 {len(files)} 个文件")
    return task_id


def update_task_progress(
    task_id: str,
    status_msg: str,
    progress_pct: int,
    files_processed: int,
    total_files: int,
) -> bool:
    """
    更新任务进度。

    Args:
        task_id: 任务ID
        status_msg: 状态描述消息
        progress_pct: 进度百分比（0-100）
        files_processed: 已处理文件数
        total_files: 总文件数

    Returns:
        bool: 更新是否成功
    """
    if task_id in tasks:
        # 记录前一个状态，用于检测状态变化
        previous_status = tasks[task_id].get("status", "")
        previous_progress = tasks[task_id].get("progress", 0)

        tasks[task_id]["status"] = status_msg
        # 确保进度值在0到100之间
        tasks[task_id]["progress"] = max(0, min(100, int(progress_pct)))
        tasks[task_id]["files_processed"] = files_processed
        tasks[task_id]["total_files"] = total_files

        # 状态变化或进度发生明显变化时才记录日志，避免日志过多
        if previous_status != status_msg or abs(previous_progress - progress_pct) >= 5:
            logger.info(
                f"任务 {task_id} 进度更新: {status_msg} - {progress_pct}% ({files_processed}/{total_files} 文件)"
            )
        return True
    else:
        # 如果任务ID未知（理论上不应发生），打印警告
        logger.warning(f"尝试为未知的任务ID更新进度: {task_id}")
        return False


def mark_task_completed(task_id: str, result_file: str = None) -> bool:
    """
    标记任务为已完成。

    Args:
        task_id: 任务ID
        result_file: 结果文件名（可选）

    Returns:
        bool: 操作是否成功
    """
    if task_id in tasks:
        tasks[task_id]["status"] = "Completed"
        tasks[task_id]["progress"] = 100
        if result_file:
            tasks[task_id]["result_file"] = result_file

        # 如果是预处理任务，记录为最近完成的预处理任务ID
        if tasks[task_id].get("task_type") == "complete_processing":
            global LATEST_COMPLETED_PREPROCESSING_TASK_ID
            LATEST_COMPLETED_PREPROCESSING_TASK_ID = task_id
            logger.info(f"已将任务 {task_id} 设置为最近完成的预处理任务")

        # 更新历史记录
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_task_history_entry(
            task_id,
            {
                "status": "成功",
                "completion_time": now,
                "processed_file_name": result_file,
            },
        )

        logger.info(f"任务 {task_id} 已标记为完成，结果文件: {result_file}")
        return True
    else:
        logger.warning(f"尝试标记未知任务 {task_id} 为完成")
        return False


def mark_task_failed(task_id: str, error_msg: str) -> bool:
    """
    标记任务为失败。

    Args:
        task_id: 任务ID
        error_msg: 错误消息

    Returns:
        bool: 操作是否成功
    """
    if task_id in tasks:
        tasks[task_id]["status"] = "Failed"
        tasks[task_id]["progress"] = 100  # 设置为100%以让前端停止轮询
        tasks[task_id]["error"] = error_msg

        # 更新历史记录
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_task_history_entry(
            task_id,
            {"status": "失败", "completion_time": now, "error_message": error_msg},
        )

        logger.error(f"任务 {task_id} 已标记为失败: {error_msg}")
        return True
    else:
        logger.warning(f"尝试标记未知任务 {task_id} 为失败")
        return False


def get_latest_task() -> Optional[str]:
    """
    获取最近一个完成的预处理任务ID。

    Returns:
        Optional[str]: 最近任务ID，如果没有则返回None
    """
    return LATEST_COMPLETED_PREPROCESSING_TASK_ID


def update_batch_task_status(
    task_id: str, batch_id: str, batch_status: str, api_response: dict = None
) -> bool:
    """
    更新批处理任务状态。

    Args:
        task_id: 任务ID
        batch_id: 批处理ID
        batch_status: 批处理状态
        api_response: API响应的完整信息（可选）

    Returns:
        bool: 更新是否成功
    """
    if task_id not in tasks:
        logger.warning(f"尝试更新不存在的任务: {task_id}")
        return False

    # 如果任务不是批处理模式，转换为批处理模式
    if not tasks[task_id].get("batch_mode"):
        tasks[task_id]["batch_mode"] = True
        tasks[task_id]["batch_id"] = batch_id
        logger.info(f"任务 {task_id} 已转换为批处理模式, 批处理ID: {batch_id}")

    # 更新批处理状态
    previous_status = tasks[task_id].get("batch_status", "")
    tasks[task_id]["batch_status"] = batch_status

    # 保存完整的API响应
    if api_response:
        tasks[task_id]["api_response"] = api_response

    # 根据状态更新任务进度
    status_progress_map = {
        "submitted": 10,
        "queued": 15,
        "running": 30,
        "processing": 50,
        "analyzed": 80,
        "completed": 95,
        "succeed": 100,
        "failed": 100,
        "error": 100,
    }

    # 如果有状态映射的进度值，更新进度
    if batch_status.lower() in status_progress_map:
        progress = status_progress_map[batch_status.lower()]
        tasks[task_id]["progress"] = progress

        # 更新状态信息
        if batch_status.lower() in ["failed", "error"]:
            tasks[task_id]["status"] = "Failed"
            error_msg = "批处理任务失败"
            if (
                api_response
                and isinstance(api_response, dict)
                and api_response.get("message")
            ):
                error_msg += f": {api_response.get('message')}"
            tasks[task_id]["error"] = error_msg
            logger.error(f"任务 {task_id} 批处理失败: {error_msg}")
        elif batch_status.lower() in ["completed", "succeed"]:
            tasks[task_id]["status"] = "BatchCompleted"
            logger.info(f"任务 {task_id} 批处理已完成，等待下载结果")
        else:
            tasks[task_id]["status"] = "BatchProcessing"

    # 只在状态变化时记录日志
    if previous_status != batch_status:
        logger.info(
            f"任务 {task_id} 批处理状态更新: {previous_status} -> {batch_status}"
        )

    return True


def add_task_history_entry(
    task_id: str, entry_type: str = "complete_processing"
) -> Dict:
    """
    添加一条新的任务历史记录。

    Args:
        task_id: 任务ID
        entry_type: 任务类型，"完整处理"或"直接导入飞书"

    Returns:
        Dict: 添加的历史记录条目
    """
    global TASK_HISTORY

    # 获取当前任务信息
    task_info = tasks.get(task_id, {})

    # 创建新的历史记录条目
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "task_id": task_id,
        "upload_time": now,
        "operation_type": (
            "完整处理" if entry_type == "complete_processing" else "直接导入飞书"
        ),
        "status": task_info.get("status", "处理中"),
        "source_files": [],
        "processed_file_name": None,
        "processed_file_path_relative": None,
        "edited_file_name": None,
        "edited_file_path_relative": None,
        "direct_import_source_file_name": None,
        "direct_import_source_file_path_relative": None,
        "completion_time": None,
        "sync_status": None,
        "error_message": task_info.get("error", ""),
    }

    # 将新记录插入到列表开头
    TASK_HISTORY.insert(0, new_entry)

    # 保持历史记录数量不超过限制
    if len(TASK_HISTORY) > MAX_HISTORY_ENTRIES:
        TASK_HISTORY = TASK_HISTORY[:MAX_HISTORY_ENTRIES]

    # 保存到文件
    save_task_history()

    logger.info(f"已添加任务 {task_id} 的历史记录，类型为 {entry_type}")
    return new_entry


def update_task_history_entry(task_id: str, updates: Dict) -> bool:
    """
    更新指定任务ID的历史记录。

    Args:
        task_id: 要更新的任务ID
        updates: 要更新的字段字典

    Returns:
        bool: 更新是否成功
    """
    global TASK_HISTORY

    # 查找匹配的历史记录条目
    for entry in TASK_HISTORY:
        if entry["task_id"] == task_id:
            # 更新提供的字段
            for key, value in updates.items():
                entry[key] = value

            # 保存更改
            save_task_history()
            logger.info(f"已更新任务 {task_id} 的历史记录")
            return True

    logger.warning(f"未找到任务 {task_id} 的历史记录，无法更新")
    return False


def load_task_history() -> List[Dict]:
    """
    从JSON文件加载任务历史记录。如果文件不存在或读取错误，则初始化为空列表。

    Returns:
        List[Dict]: 任务历史记录列表
    """
    global TASK_HISTORY
    try:
        if os.path.exists(HISTORY_FILE_PATH):
            with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f:
                data = f.read().strip()
                if not data:  # 文件为空
                    logger.warning(
                        f"历史记录文件 {HISTORY_FILE_PATH} 为空，初始化为空列表"
                    )
                    TASK_HISTORY = []
                    return TASK_HISTORY

                # 修复：使用json.loads(data)而不是json.load(f)
                # 因为f.read()已经将指针移到了文件末尾
                try:
                    TASK_HISTORY = json.loads(data)
                    logger.info(
                        f"已从 {HISTORY_FILE_PATH} 加载 {len(TASK_HISTORY)} 条历史记录"
                    )

                    # 验证数据结构
                    if not isinstance(TASK_HISTORY, list):
                        logger.warning(f"历史记录文件格式错误，重置为空列表")
                        TASK_HISTORY = []
                except json.JSONDecodeError as e:
                    logger.error(f"历史记录文件JSON解析错误: {e}，重置为空列表")
                    TASK_HISTORY = []
        else:
            logger.info(f"历史记录文件不存在，将初始化为空列表")
            TASK_HISTORY = []
    except Exception as e:
        logger.error(f"加载历史记录时出错: {e}", exc_info=True)
        TASK_HISTORY = []

    return TASK_HISTORY


def save_task_history() -> bool:
    """
    将当前任务历史记录保存到JSON文件。

    Returns:
        bool: 保存是否成功
    """
    try:
        with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(TASK_HISTORY, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存 {len(TASK_HISTORY)} 条历史记录到 {HISTORY_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"保存历史记录时出错: {e}")
        return False


def clean_up_old_tasks(days_to_keep: int = 7) -> int:
    """
    清理过期的任务和相关文件。

    Args:
        days_to_keep: 保留的天数，默认为7天

    Returns:
        int: 清理的任务数量
    """
    # 计算超时日期
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    removed_count = 0

    # 从历史记录查找过期任务
    global TASK_HISTORY
    tasks_to_remove = []

    logger.info(f"开始清理过期任务 (保留 {days_to_keep} 天内的任务)")

    for entry in TASK_HISTORY:
        try:
            # 获取上传时间或完成时间
            time_str = entry.get("completion_time") or entry.get("upload_time")
            if not time_str:
                continue

            task_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

            # 检查是否过期
            if task_time < cutoff_date:
                task_id = entry.get("task_id")
                if task_id:
                    tasks_to_remove.append(task_id)
                    logger.debug(f"任务 {task_id} 已过期 (创建于 {time_str})")

        except (ValueError, TypeError) as e:
            logger.warning(f"解析任务时间时出错: {e}")

    if not tasks_to_remove:
        logger.info("没有找到需要清理的过期任务")
        return 0

    logger.info(f"找到 {len(tasks_to_remove)} 个过期任务需要清理")

    # 清理过期任务
    for task_id in tasks_to_remove:
        try:
            # 从内存中删除任务状态
            if task_id in tasks:
                del tasks[task_id]
                logger.debug(f"已从内存中删除任务状态: {task_id}")

            # 删除任务目录（如果存在）
            upload_dir = os.path.join("uploads", task_id)
            output_dir = os.path.join("outputs", task_id)

            # 删除上传目录
            if os.path.exists(upload_dir) and os.path.isdir(upload_dir):
                file_count = len(os.listdir(upload_dir))
                for filename in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            logger.debug(f"已删除文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件 {file_path} 时出错: {e}")
                try:
                    os.rmdir(upload_dir)
                    logger.debug(
                        f"已删除上传目录: {upload_dir} (包含 {file_count} 个文件)"
                    )
                except Exception as e:
                    logger.error(f"删除目录 {upload_dir} 时出错: {e}")

            # 删除输出目录
            if os.path.exists(output_dir) and os.path.isdir(output_dir):
                file_count = len(os.listdir(output_dir))
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            logger.debug(f"已删除文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件 {file_path} 时出错: {e}")
                try:
                    os.rmdir(output_dir)
                    logger.debug(
                        f"已删除输出目录: {output_dir} (包含 {file_count} 个文件)"
                    )
                except Exception as e:
                    logger.error(f"删除目录 {output_dir} 时出错: {e}")

            removed_count += 1
            logger.info(f"已清理过期任务 {task_id} 及其文件")

        except Exception as e:
            logger.error(f"清理任务 {task_id} 时出错: {e}", exc_info=True)

    # 从历史记录中过滤掉已删除的任务
    if removed_count > 0:
        original_count = len(TASK_HISTORY)
        TASK_HISTORY = [
            entry
            for entry in TASK_HISTORY
            if entry.get("task_id") not in tasks_to_remove
        ]
        logger.info(
            f"已从历史记录中删除 {original_count - len(TASK_HISTORY)} 个过期任务记录"
        )
        save_task_history()

    logger.info(f"已清理 {removed_count} 个过期任务 (保留近 {days_to_keep} 天内的任务)")
    return removed_count


# 初始化加载历史记录
load_task_history()
