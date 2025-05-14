from flask import Blueprint, jsonify, request, current_app
import os
import json
import traceback
import threading
import time  # 新增导入time模块
from utils.error_handler import handle_exceptions
from utils.task_manager import (
    get_task_info,
    update_task_progress,
    mark_task_completed,
    mark_task_failed,
    load_task_history,
)
import processor
import postprocessor
import feishu_utils
import pandas as pd
from utils.logger import setup_logger

# 创建Blueprint
task_bp = Blueprint("task", __name__)

# 添加全局变量存储后台查询线程，避免线程被垃圾回收
background_checking_threads = {}

# 添加线程锁，防止并发处理同一任务
processing_locks = {}

logger = setup_logger("task_routes")


# 新增函数：自动周期性查询百炼任务状态
def auto_check_batch_status(task_id, app, interval=60):
    """
    自动定期查询百炼任务状态的后台线程函数

    Args:
        task_id: 任务ID
        app: Flask应用实例
        interval: 查询间隔（秒），默认为60秒
    """
    # 使用传入的app创建应用上下文
    with app.app_context():
        logger.info(
            f"启动自动查询百炼任务状态线程: 任务ID={task_id}, 间隔={interval}秒"
        )

        while True:
            try:
                # 获取任务信息
                task_info = get_task_info(task_id)
                if not task_info:
                    logger.warning(f"任务已不存在，停止自动查询: {task_id}")
                    break

                # 如果任务已经完成或失败，则停止查询
                if task_info.get("status") in ["completed", "failed", "error"]:
                    logger.info(f"任务已完成或失败，停止自动查询: {task_id}")
                    # 移除线程引用
                    if task_id in background_checking_threads:
                        del background_checking_threads[task_id]
                    break

                # 检查是否是百炼批处理任务
                if not task_info.get("batch_mode") or not task_info.get("batch_id"):
                    logger.warning(f"不是百炼批处理任务，停止自动查询: {task_id}")
                    break

                logger.info(f"自动查询百炼任务状态: {task_id}")
                batch_id = task_info.get("batch_id")

                # 查询批处理任务状态
                status_info = processor.check_bailian_job_status(
                    batch_id, task_info.get("config", {})
                )
                batch_status = status_info.get("status", "")

                logger.info(f"百炼任务状态: {batch_status}, 任务ID: {task_id}")

                # 更新任务状态
                if batch_status == "completed":
                    logger.info(f"百炼任务已完成，开始处理结果: {task_id}")
                    # 百炼任务已完成，复用check_batch_status中的处理逻辑
                    # 更新任务状态并触发结果处理
                    process_completed_batch(task_id, task_info, app)

                    # 处理已启动，可以终止自动查询线程
                    if task_id in background_checking_threads:
                        del background_checking_threads[task_id]
                    break

                elif batch_status in ["failed", "expired", "canceled"]:
                    # 百炼任务失败
                    error_msg = f"百炼批处理异常终止: {batch_status}"
                    mark_task_failed(task_id, error_msg)

                    # 移除线程引用
                    if task_id in background_checking_threads:
                        del background_checking_threads[task_id]
                    break

                else:
                    # 百炼任务仍在处理中，更新进度消息
                    progress_msg = f"百炼批处理中: {batch_status}"
                    logger.info(f"{progress_msg}, 任务ID: {task_id}")
                    update_task_progress(
                        task_id,
                        progress_msg,
                        80,
                        task_info.get("total_files", 1),
                        task_info.get("total_files", 1),
                    )

            except Exception as e:
                logger.error(f"自动查询百炼任务状态时出错: {e}", exc_info=True)
                traceback.print_exc()

                # 如果出错，尝试继续查询而不是终止线程
                # 但是记录错误
                update_task_progress(
                    task_id,
                    f"查询百炼任务状态出错: {str(e)[:50]}...",
                    70,
                    task_info.get("total_files", 1),
                    task_info.get("total_files", 1),
                )

            # 等待下一次查询
            time.sleep(interval)


# 修改process_completed_batch函数
def process_completed_batch(task_id, task_info, app=None):
    """
    处理已完成的百炼批处理任务，下载和处理结果

    Args:
        task_id: 任务ID
        task_info: 任务信息字典
        app: Flask应用实例，如果为None则尝试从current_app获取
    """
    global processing_locks

    # 获取或创建任务专用锁
    if task_id not in processing_locks:
        processing_locks[task_id] = threading.Lock()

    # 尝试获取锁，如果已被占用则表示该任务已经在处理中
    if not processing_locks[task_id].acquire(blocking=False):
        logger.warning(f"任务 {task_id} 已有另一个线程在处理，跳过本次处理")
        return

    try:
        # 获取正确的app实例
        if app is None:
            try:
                app = current_app._get_current_object()
            except RuntimeError:
                logger.error(f"处理百炼结果时无法获取应用上下文，任务ID: {task_id}")
                mark_task_failed(task_id, "处理百炼结果时无法获取应用上下文")
                return

        # 更新状态但不标记为完成
        update_task_progress(
            task_id,
            "百炼批处理已完成，正在处理结果",
            90,
            task_info.get("total_files", 1),
            task_info.get("total_files", 1),
        )

        # 检查任务是否已经处理过，避免重复处理
        fresh_task_info = get_task_info(task_id)
        if fresh_task_info and fresh_task_info.get("result_file"):
            logger.info(f"任务 {task_id} 已有结果文件，跳过处理")
            return

        # 启动后台线程处理结果，避免阻塞当前线程
        process_thread = threading.Thread(
            target=process_batch_result, args=(task_id, task_info, app), daemon=True
        )
        process_thread.start()

        # 等待处理线程完成，最多等待10分钟
        process_thread.join(timeout=600)

        # 检查处理是否完成
        if process_thread.is_alive():
            logger.warning(f"任务 {task_id} 处理超时(10分钟)，但处理线程仍在继续运行")
    finally:
        # 释放锁
        processing_locks[task_id].release()

        # 清理过多的锁，避免内存泄漏
        if len(processing_locks) > 100:
            # 只保留正在使用的锁
            active_locks = {}
            for tid, lock in processing_locks.items():
                if lock.locked():
                    active_locks[tid] = lock
            processing_locks = active_locks


# 修改现有的run_processing函数，在百炼模式下自动启动状态查询
def run_processing(task_id, input_files, output_file, config):
    """
    在后台线程中运行的包装函数，调用processor的主处理函数并更新任务状态。

    Args:
        task_id: 任务ID
        input_files: 输入文件路径列表
        output_file: 输出文件路径
        config: 配置字典
    """
    try:
        # 定义进度回调函数
        def progress_callback(status_msg, progress_pct, files_processed, total_files):
            update_task_progress(
                task_id, status_msg, progress_pct, files_processed, total_files
            )

        # 调用processor模块的核心处理函数
        result = processor.process_files_and_consolidate(
            input_files,
            output_file,
            config,  # 直接传递完整的config字典
            progress_callback,
        )

        # 更新任务状态
        if result:
            # 检查返回值类型，处理百炼API和普通处理模式的不同返回值
            if isinstance(result, tuple) and len(result) == 2:
                # 百炼API模式 - 返回(batch_id, task_id)元组
                batch_id, bailian_task_id = result

                # 存储百炼批次信息到任务状态中
                task = get_task_info(task_id)
                if task:
                    task["batch_id"] = batch_id
                    task["bailian_task_id"] = bailian_task_id
                    task["status"] = "BatchProcessing"
                    task["batch_mode"] = True

                # 设置任务状态
                update_task_progress(
                    task_id,
                    "百炼批处理中",
                    70,
                    len(input_files),
                    len(input_files),
                )

                # 注意：不在这里启动自动查询线程，让全局批处理任务检查功能来处理
                # 这样可以避免应用上下文问题，并确保所有批处理任务都能被检查
                logger.info(
                    f"任务 {task_id} 已提交到百炼批处理，将由全局检查线程自动监控状态"
                )

            else:
                # 常规模式 - 返回结果文件路径字符串
                result_file = result
                # 标记任务完成
                mark_task_completed(task_id, os.path.basename(result_file))
        else:
            # 处理失败
            mark_task_failed(task_id, "处理失败，未生成结果文件。")

    except Exception as e:
        # 处理过程中出现异常
        error_msg = str(e)
        traceback.print_exc()
        mark_task_failed(task_id, error_msg)


@task_bp.route("/progress/<task_id>")
@handle_exceptions
def progress(task_id):
    """
    查询指定任务ID的处理进度和状态。

    前端通过定期轮询此API获取任务进度，进度信息包括百分比完成度、
    当前处理阶段描述、已处理文件数量等。当任务完成时会返回结果文件名，
    用于前端生成下载链接。如果任务ID不存在则返回404错误。

    Path Params:
        - task_id: 任务唯一标识符UUID

    Returns:
        JSON: 包含任务状态、进度百分比、处理文件信息和结果文件名(如果完成)
    """
    task = get_task_info(task_id)
    if not task:
        # 如果任务ID不存在（可能已完成很久被清理，或从未存在）
        # 返回404错误，并提供一个表示任务未知或结束的状态
        return (
            jsonify(
                {
                    "status": "未知或已过期的任务",
                    "progress": 100,  # 让前端停止轮询
                    "error": "任务ID未找到。",
                }
            ),
            404,
        )
    # 返回当前任务的状态信息
    return jsonify(task)


@task_bp.route("/api/history")
@handle_exceptions
def get_history():
    """
    获取任务历史记录的API接口。

    Returns:
        JSON: 包含最近的任务历史记录列表
    """
    # 重新加载历史记录以确保最新数据
    history_data = load_task_history()
    return jsonify(history_data)


@task_bp.route("/check_batch_status/<task_id>", methods=["GET"])
@handle_exceptions
def check_batch_status(task_id):
    """
    查询百炼批处理任务的状态。

    当任务使用百炼批处理API时，此路由用于查询批处理任务的当前状态，
    判断是否已经完成，并在完成时触发后续的结果下载和处理步骤。

    Path Params:
        - task_id: 任务唯一标识符UUID

    Returns:
        JSON: 批处理任务的状态信息和进度
    """
    task_info = get_task_info(task_id)
    if not task_info:
        return jsonify({"success": False, "error": "任务ID不存在或已过期"}), 404

    # 检查是否是百炼批处理任务
    if not task_info.get("batch_mode") or not task_info.get("batch_id"):
        return jsonify({"success": False, "error": "不是百炼批处理任务"}), 400

    batch_id = task_info.get("batch_id")

    try:
        # 查询批处理任务状态
        status_info = processor.check_bailian_job_status(
            batch_id, task_info.get("config", {})
        )
        batch_status = status_info.get("status", "")

        # 更新任务状态
        if batch_status == "completed":
            # 百炼任务已完成，更新状态并触发结果处理
            # 获取当前Flask应用实例
            app = current_app._get_current_object()
            # 使用修改后的函数，传递app对象
            process_completed_batch(task_id, task_info, app)

            return jsonify(
                {
                    "success": True,
                    "status": "completed",
                    "message": "百炼批处理已完成，正在处理结果",
                }
            )

        elif batch_status in ["failed", "expired", "canceled"]:
            # 百炼任务失败
            error_msg = f"百炼批处理异常终止: {batch_status}"
            mark_task_failed(task_id, error_msg)

            return jsonify(
                {"success": False, "status": batch_status, "message": error_msg}
            )

        else:
            # 百炼任务仍在处理中
            progress_msg = f"百炼批处理中: {batch_status}"
            task_info["batch_status"] = batch_status

            # 更新进度消息
            update_task_progress(
                task_id,
                progress_msg,
                80,
                task_info.get("total_files", 1),
                task_info.get("total_files", 1),
            )

            return jsonify(
                {
                    "success": True,
                    "status": batch_status,
                    "message": progress_msg,
                    "created_at": status_info.get("created_at"),
                    "expires_at": status_info.get("expires_at"),
                }
            )

    except Exception as e:
        error_msg = f"查询百炼批处理状态时出错: {e}"
        mark_task_failed(task_id, error_msg)
        return jsonify({"success": False, "error": error_msg}), 500


# 修改check_all_batch_tasks函数
def check_all_batch_tasks(app, interval=60):
    """
    定时检查所有处于批处理状态的任务

    Args:
        app: Flask应用实例
        interval: 检查间隔（秒），默认为60秒
    """
    logger.info(f"启动全局百炼批处理任务检查线程，检查间隔: {interval}秒")

    # 记录是否正在运行检查，避免重复检查
    checking_in_progress = False

    # 跟踪已处理的任务，防止重复处理
    processed_tasks = set()

    while True:
        try:
            # 只有在非检查状态时才执行检查
            if not checking_in_progress:
                checking_in_progress = True

                # 使用应用上下文
                with app.app_context():
                    logger.info(f"执行全局百炼批处理任务检查...")

                    # 加载所有任务历史
                    history_data = load_task_history()

                    # 遍历最近的任务，寻找批处理任务
                    for task_item in history_data:
                        task_id = task_item.get("task_id")

                        # 跳过已经处理过的任务
                        if task_id in processed_tasks:
                            continue

                        # 获取任务详情
                        task_info = get_task_info(task_id)
                        if not task_info:
                            continue

                        # 只处理批处理模式的任务
                        if not task_info.get("batch_mode") or not task_info.get(
                            "batch_id"
                        ):
                            continue

                        # 增强对已完成任务的判断：
                        # 1. 检查status字段
                        # 2. 检查是否有result_file（表示已完成）
                        # 3. 检查progress是否为100（完成状态）
                        if (
                            task_info.get("status") in ["completed", "failed", "error"]
                            or task_info.get("result_file") is not None
                            or task_info.get("progress", 0) >= 100
                        ):
                            # 记录为已处理，避免将来重复检查
                            processed_tasks.add(task_id)
                            continue

                        # 查询批处理任务状态
                        batch_id = task_info.get("batch_id")
                        logger.info(
                            f"全局检查百炼批处理任务: {task_id}, 批次ID: {batch_id}"
                        )

                        try:
                            status_info = processor.check_bailian_job_status(
                                batch_id, task_info.get("config", {})
                            )
                            batch_status = status_info.get("status", "")

                            logger.info(
                                f"百炼任务状态: {batch_status}, 任务ID: {task_id}"
                            )

                            # 根据状态处理任务
                            if batch_status == "completed":
                                logger.info(f"百炼任务已完成，开始处理结果: {task_id}")
                                # 处理已完成的任务
                                process_completed_batch(
                                    task_id, task_info, app
                                )  # 传递app对象

                                # 标记任务已处理，避免重复处理
                                processed_tasks.add(task_id)

                            elif batch_status in ["failed", "expired", "canceled"]:
                                # 百炼任务失败
                                error_msg = f"百炼批处理异常终止: {batch_status}"
                                mark_task_failed(task_id, error_msg)

                                # 标记任务已处理，避免重复处理
                                processed_tasks.add(task_id)

                            else:
                                # 百炼任务仍在处理中
                                progress_msg = f"百炼批处理中: {batch_status}"
                                update_task_progress(
                                    task_id,
                                    progress_msg,
                                    80,
                                    task_info.get("total_files", 1),
                                    task_info.get("total_files", 1),
                                )
                        except Exception as e:
                            logger.error(f"检查任务 {task_id} 状态时出错: {e}")
                            traceback.print_exc()

                            # 记录错误但继续检查其他任务
                            update_task_progress(
                                task_id,
                                f"查询百炼任务状态出错: {str(e)[:50]}...",
                                task_info.get("progress", 70),
                                task_info.get("files_processed", 0),
                                task_info.get("total_files", 1),
                            )

                    # 清理过旧的已处理任务记录，只保留最近100个
                    if len(processed_tasks) > 100:
                        processed_tasks = set(list(processed_tasks)[-100:])

                    logger.info(
                        f"全局百炼批处理任务检查完成，将在 {interval} 秒后再次检查"
                    )

                # 检查完成，重置标志
                checking_in_progress = False

        except Exception as e:
            logger.error(f"全局百炼批处理任务检查线程出错: {e}")
            traceback.print_exc()
            # 检查失败，重置标志
            checking_in_progress = False

        # 等待下一次查询
        time.sleep(interval)


# 修改process_batch_result函数，正确处理合并数据
def process_batch_result(task_id, task_info, app):
    """
    处理百炼批处理任务的结果，并将结果与飞书数据合并

    Args:
        task_id: 任务ID
        task_info: 任务信息字典
        app: Flask应用实例
    """
    try:
        # 使用应用上下文
        with app.app_context():
            logger.info(f"开始处理百炼批处理结果并与飞书数据合并，任务ID: {task_id}")

            # 检查任务是否已经处理过，避免重复处理
            # 重新获取最新任务状态
            fresh_task_info = get_task_info(task_id)
            if fresh_task_info:
                # 检查任务是否已经处理完成
                if (
                    fresh_task_info.get("status") in ["completed", "failed", "error"]
                    or fresh_task_info.get("result_file") is not None
                    or fresh_task_info.get("progress", 0) >= 100
                ):
                    logger.info(f"警告: 任务 {task_id} 已经处理过，跳过重复处理")
                    return

            # 获取输出目录
            output_dir = os.path.join(app.config["OUTPUT_FOLDER"], task_id)

            # 检查是否已经存在结果文件，防止重复处理
            result_file_pattern = f"final_{task_id}.xlsx"
            simple_result_pattern = f"simple_results_{task_id}.xlsx"

            if os.path.exists(
                os.path.join(output_dir, result_file_pattern)
            ) or os.path.exists(os.path.join(output_dir, simple_result_pattern)):
                logger.info(
                    f"检测到结果文件已存在，任务 {task_id} 可能已经处理过，跳过重复处理"
                )
                # 确保任务状态标记为已完成
                if not fresh_task_info.get("result_file"):
                    if os.path.exists(os.path.join(output_dir, result_file_pattern)):
                        mark_task_completed(task_id, result_file_pattern)
                    else:
                        mark_task_completed(task_id, simple_result_pattern)
                return

            update_task_progress(
                task_id,
                "下载和处理百炼批处理结果中",
                90,
                task_info.get("total_files", 1),
                task_info.get("total_files", 1),
            )

            # 1. 下载和处理百炼批处理结果
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"输出目录: {output_dir}")

            # 构造输出文件路径
            consolidated_output = os.path.join(
                output_dir, f"consolidated_{task_id}.xlsx"
            )

            # 从任务信息中获取批次ID
            batch_id = task_info.get("batch_id")
            logger.info(f"批次ID: {batch_id}")

            # 从任务信息中获取配置
            config = task_info.get("config", {})

            # 下载并处理百炼结果
            logger.info("正在下载和处理百炼结果...")
            bailian_df = processor.download_and_process_bailian_results(
                batch_id,
                None,  # 不再传递原始DataFrame
                task_id,  # 任务ID
                config,
            )

            if bailian_df is None or bailian_df.empty:
                logger.warning("警告: 百炼结果处理后返回了空的 DataFrame")
                error_msg = "百炼API结果处理失败，未生成有效数据。"
                mark_task_failed(task_id, error_msg)
                return

            logger.info(f"百炼API结果下载完成，获得 {len(bailian_df)} 行数据")

            # 2. 从飞书获取数据
            logger.info("开始从飞书获取数据...")
            feishu_config = config.get("feishu_config", {})
            feishu_df = None

            try:
                # 验证飞书配置是否完整
                required_config = ["APP_ID", "APP_SECRET", "APP_TOKEN", "TABLE_IDS"]
                missing_config = [
                    key for key in required_config if not feishu_config.get(key)
                ]

                if missing_config:
                    logger.warning(
                        f"警告: 飞书配置缺少必要参数: {missing_config}，跳过飞书数据获取"
                    )
                else:
                    # 调用feishu_utils.fetch_and_prepare_feishu_data获取数据
                    logger.info(
                        f"开始获取飞书数据，表格IDs: {feishu_config.get('TABLE_IDS')}"
                    )
                    # 获取目标列配置
                    target_columns = config.get("llm_config", {}).get(
                        "TARGET_COLUMNS", []
                    )
                    feishu_df = feishu_utils.fetch_and_prepare_feishu_data(
                        feishu_config, target_columns
                    )

                    if feishu_df is not None and not feishu_df.empty:
                        logger.info(f"从飞书获取了 {len(feishu_df)} 行数据")
                    else:
                        logger.info("从飞书获取数据为空")
            except Exception as e:
                logger.error(f"从飞书获取数据时出错: {e}")
                traceback.print_exc()
                logger.info("将继续处理，仅使用百炼API结果")

            # 3. 合并数据集
            logger.info("开始合并数据集...")

            # 保存原始百炼数据
            original_bailian_df = bailian_df.copy()

            # 如果获取了飞书数据，则合并
            if feishu_df is not None and not feishu_df.empty:
                logger.info(
                    f"合并 {len(bailian_df)} 行百炼数据和 {len(feishu_df)} 行飞书数据"
                )

                # 确保系统列存在
                for col in ["来源", "record_id"]:
                    if col not in bailian_df.columns:
                        bailian_df[col] = ""
                    if col not in feishu_df.columns:
                        feishu_df[col] = ""

                # === 新增：补全来源字段 ===
                # 本地数据：空值补文件名
                file_name = task_info.get("file_name", "本地文件")
                mask_local = (bailian_df["来源"] == "") | (bailian_df["来源"].isna())
                bailian_df.loc[mask_local, "来源"] = file_name
                logger.info(
                    f"本地数据'来源'字段补全为文件名: {file_name}，补全行数: {mask_local.sum()}"
                )

                # 飞书数据：空值补"飞书"
                mask_feishu = (feishu_df["来源"] == "") | (feishu_df["来源"].isna())
                feishu_df.loc[mask_feishu, "来源"] = "飞书"
                logger.info(
                    f"飞书数据'来源'字段补全为'飞书'，补全行数: {mask_feishu.sum()}"
                )

                # 合并数据集，简单的垂直合并
                combined_df = pd.concat([feishu_df, bailian_df], ignore_index=True)
                logger.info(f"数据合并完成，共 {len(combined_df)} 行")
                # 日志：来源字段唯一值与缺失统计
                logger.info(
                    f"合并后'来源'字段唯一值: {combined_df['来源'].unique().tolist()}"
                )
                logger.info(
                    f"合并后'来源'字段缺失行数: {(combined_df['来源'] == '').sum() + combined_df['来源'].isna().sum()}"
                )

                # 保存合并的原始数据用于调试
                raw_output = os.path.join(output_dir, f"raw_combined_{task_id}.csv")
                combined_df.to_csv(raw_output, index=False)
                logger.info(f"已保存原始合并数据到 {raw_output}")

                # 保存原始合并数据（这个是飞书数据+百炼API结果的合并，作为原始数据）
                raw_combined_df = combined_df.copy()

                # 应用后处理步骤
                logger.info(
                    f"应用后处理步骤，合并DataFrame大小: {len(combined_df)} 行..."
                )
                processed_df, df_new, df_update, update_ids = (
                    postprocessor.apply_post_processing(
                        combined_df, config, id_column="行ID"
                    )
                )
            else:
                logger.info("没有可用的飞书数据，仅使用百炼API结果")
                # 本地数据：空值补文件名
                file_name = task_info.get("file_name", "本地文件")
                mask_local = (bailian_df["来源"] == "") | (bailian_df["来源"].isna())
                bailian_df.loc[mask_local, "来源"] = file_name
                logger.info(
                    f"本地数据'来源'字段补全为文件名: {file_name}，补全行数: {mask_local.sum()}"
                )
                # 日志：来源字段唯一值与缺失统计
                logger.info(
                    f"仅本地数据'来源'字段唯一值: {bailian_df['来源'].unique().tolist()}"
                )
                logger.info(
                    f"仅本地数据'来源'字段缺失行数: {(bailian_df['来源'] == '').sum() + bailian_df['来源'].isna().sum()}"
                )

                raw_combined_df = original_bailian_df.copy()
                logger.info(f"应用后处理步骤，DataFrame大小: {len(bailian_df)} 行...")
                processed_df, df_new, df_update, update_ids = (
                    postprocessor.apply_post_processing(
                        bailian_df, config, id_column="行ID"
                    )
                )

            # 4. 创建多Sheet页Excel文件
            if not processed_df.empty:
                final_output_file = os.path.join(output_dir, f"final_{task_id}.xlsx")
                logger.info(f"创建多Sheet页Excel文件: {final_output_file}")

                try:
                    # 保存原始数据备份，以防create_multi_sheet_excel函数失败
                    temp_csv_path = os.path.join(
                        output_dir, f"processed_data_backup_{task_id}.csv"
                    )
                    processed_df.to_csv(temp_csv_path, index=False)
                    logger.info(f"已保存处理后数据备份到 {temp_csv_path}")

                    # 使用增强的multi-sheet excel创建函数，并传递原始合并数据
                    postprocessor.create_multi_sheet_excel(
                        processed_df,
                        final_output_file,
                        config,
                        raw_combined_df,  # 传递原始数据，确保"原始数据"Sheet正确
                        id_column="行ID",  # 使用行ID作为唯一标识符
                        update_ids=update_ids,  # 修复：补充 update_ids 参数
                        df_new=df_new,  # 新增Sheet数据
                        df_update=df_update,  # 更新Sheet数据
                    )

                    logger.info(f"✅ 结果已保存到 {final_output_file}")

                    # 标记任务完成
                    mark_task_completed(task_id, f"final_{task_id}.xlsx")
                    logger.info(f"✅ 任务 {task_id} 已标记为完成")
                except Exception as e:
                    logger.error(f"创建多Sheet页Excel文件时出错: {e}")
                    traceback.print_exc()

                    # 尝试只保存单一sheet的Excel
                    try:
                        simple_excel_path = os.path.join(
                            output_dir, f"simple_results_{task_id}.xlsx"
                        )
                        processed_df.to_excel(simple_excel_path, index=False)
                        logger.info(f"已创建简单Excel文件作为备份: {simple_excel_path}")
                        mark_task_completed(task_id, f"simple_results_{task_id}.xlsx")
                    except:
                        # 如果Excel也失败，尝试保存为CSV
                        csv_path = os.path.join(output_dir, f"results_{task_id}.csv")
                        processed_df.to_csv(csv_path, index=False)
                        logger.info(f"已创建CSV文件作为备份: {csv_path}")
                        mark_task_completed(task_id, f"results_{task_id}.csv")
            else:
                # 处理失败
                error_msg = "数据处理后为空，未生成有效数据。"
                logger.error(f"错误: {error_msg}")
                mark_task_failed(task_id, error_msg)

    except Exception as e:
        logger.error(f"处理百炼批处理结果时出错: {e}")
        traceback.print_exc()
        mark_task_failed(task_id, f"处理百炼批处理结果时出错: {str(e)}")


# 修改register_routes函数，直接使用Flask应用实例而非获取代理对象
def register_routes(app):
    """
    注册所有路由到Flask应用。

    Args:
        app: Flask应用实例
    """
    app.register_blueprint(task_bp)

    # 创建应用实例属性存储线程状态
    if not hasattr(app, "_global_checker_thread"):
        app._global_checker_thread = None

    # 修改before_request函数，只启动一次全局检查线程
    @app.before_request
    def before_request_func():
        # 检查线程是否已存在且运行中
        if (
            app._global_checker_thread is None
            or not app._global_checker_thread.is_alive()
        ):
            logger.info("准备启动全局百炼批处理任务检查线程...")

            try:
                # 直接使用Flask应用实例，不再获取代理对象
                # 启动全局批处理任务检查线程
                checker_thread = threading.Thread(
                    target=check_all_batch_tasks,
                    args=(app, 60),  # 每60秒检查一次
                    daemon=True,
                )
                checker_thread.start()

                # 保存线程引用
                app._global_checker_thread = checker_thread
                logger.info("全局百炼批处理任务检查线程已启动")

            except Exception as e:
                logger.error(f"启动全局批处理任务检查线程失败: {e}")
                traceback.print_exc()
