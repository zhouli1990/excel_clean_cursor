import os
import sys
import uuid
import json
import time
import logging
import tempfile
import traceback
import threading
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    send_from_directory,
    url_for,
    flash,
    redirect,
)
import shutil
import postprocessor
import processor
import feishu_utils

# 定义配置文件路径
CONFIG_FILE = "config.json"

# 初始化 Flask 应用
app = Flask(__name__)

# --- 设置 Secret Key --- #
# 对于生产环境，强烈建议从环境变量或配置文件读取一个固定且保密的密钥
# 例如: app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_dev_key')
# 为了方便本地运行，这里使用 os.urandom 生成一个随机密钥
# 注意：每次重启服务密钥都会改变，这会导致之前的 session 失效
app.secret_key = os.urandom(24)
print(f"Flask app secret key set randomly.")

# 配置上传文件存储路径
app.config["UPLOAD_FOLDER"] = "uploads"
# 配置处理结果输出路径
app.config["OUTPUT_FOLDER"] = "outputs"
# 确保上传和输出目录存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# 使用字典在内存中存储任务状态和进度 (简单实现)
# 注意：在生产环境中，建议使用更健壮的方案，如 Redis 或数据库
tasks = {}

# --- 数据处理模块的配置 (现在作为后备默认值) ---
# 主要配置将从前端请求中获取
DEFAULT_PROCESSOR_CONFIG = {
    # 定义最终输出 Excel 文件需要包含的目标列名
    "TARGET_COLUMNS": [
        "公司名称",
        "联系人",
        "职位",
        "电话",
        "来源",  # *** 新增：将"来源"添加到默认目标列 ***
        # "邮箱",
        # "地址",
    ],
    # 从环境变量获取 DeepSeek API Key，如果未设置则使用默认值 (仅供示例)
    "DEEPSEEK_API_KEY": os.environ.get(
        "DEEPSEEK_API_KEY",
        "sk-9df835c781904b2289663567d05c94d9",  # <<<--- 从环境变量获取或直接替换
    ),
    # DeepSeek API 的接入点
    "DEEPSEEK_API_ENDPOINT": "https://api.deepseek.com/chat/completions",
    # 调用 LLM API 时，每次处理的行数 (批处理大小)
    "BATCH_SIZE": 160,  # 可根据需要调整
    # LLM API 调用时，允许生成的最大 token 数量
    "MAX_COMPLETION_TOKENS": 8192,  # 可根据 BATCH_SIZE 和模型限制调整
    "API_TIMEOUT": 180,  # 默认API超时
}

# --- 飞书配置 (后备默认值) ---
DEFAULT_FEISHU_CONFIG = {
    "APP_ID": "cli_a36634dc16b8d00e",
    "APP_SECRET": "RoXYTnSBGGsLLyvONbSCYe15Jm6bv5Xn",
    "APP_TOKEN": "XyUFbxc8JaDkTJscEigcbkxgnqe",
    "TABLE_IDS": [
        "tblEGrUKq8KKOPAc",
        "tbltHzIYuD95qhGv",
        "tbl94LyQMqtz0X45",
        "tblJPUJJEQw0VEYz",
    ],
    "ADD_TARGET_TABLE_IDS": [],  # 新增：用于添加记录的特定 Table ID 列表 (可选)
    "COMPANY_NAME_COLUMN": "企业名称",
    "PHONE_NUMBER_COLUMN": "电话",
    "REMARK_COLUMN_NAME": "备注",
    "RELATED_COMPANY_COLUMN_NAME": "关联公司名称(LLM)",  # 定义新列名
}

# --- 后处理默认列配置 (新) ---
DEFAULT_POST_PROCESSING_COLS = {
    "check_duplicate_phones": {
        "post_phone_col_for_dup_phones": "电话"  # 默认尝试使用"电话"列
    },
    "check_duplicate_companies": {
        "post_phone_col_for_dup_comp": "电话",  # 默认尝试使用"电话"列
        "post_company_col_for_dup_comp": "公司名称",  # 默认尝试使用"公司名称"列
    },
}


# --- 加载/初始化配置 ---
def load_config():
    """尝试从 config.json 加载配置，如果失败则使用默认值并保存。"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_from_file = json.load(f)
                # 简单验证一下结构
                if (
                    "llm_config" in config_from_file
                    and "feishu_config" in config_from_file
                    # 也要检查新的 post_processing_config (为了保存) 和 defaults
                    and "post_processing_config" in config_from_file
                    # and "feishu_config" in config_from_file and "ADD_TARGET_TABLE_ID" in config_from_file["feishu_config"] # Add check for new key? Optional.
                ):
                    print(f"成功从 {CONFIG_FILE} 加载配置。")
                    # 确保即使旧配置文件没有 ADD_TARGET_TABLE_ID 也能正常加载
                    loaded_feishu_config = config_from_file["feishu_config"]
                    if "ADD_TARGET_TABLE_IDS" not in loaded_feishu_config:
                        loaded_feishu_config["ADD_TARGET_TABLE_IDS"] = (
                            DEFAULT_FEISHU_CONFIG["ADD_TARGET_TABLE_IDS"]
                        )

                    # 返回加载的配置
                    return (
                        config_from_file["llm_config"],
                        loaded_feishu_config,  # 使用处理过的 feishu_config
                        config_from_file.get(
                            "post_processing_config",
                            DEFAULT_POST_PROCESSING_COLS.copy(),
                        ),  # 加载保存的后处理默认值
                    )
                else:
                    print(f"警告: {CONFIG_FILE} 文件格式不完整，使用默认配置。")
        else:
            print(f"配置文件 {CONFIG_FILE} 不存在，使用默认配置。")
    except (json.JSONDecodeError, IOError) as e:
        print(f"加载配置文件 {CONFIG_FILE} 时出错: {e}，使用默认配置。")

    # 如果加载失败或文件不存在，则使用默认值并尝试保存
    default_config = {
        "llm_config": DEFAULT_PROCESSOR_CONFIG.copy(),
        "feishu_config": DEFAULT_FEISHU_CONFIG.copy(),
        "post_processing_config": DEFAULT_POST_PROCESSING_COLS.copy(),  # 保存时也包含后处理默认值
    }
    save_config(default_config)  # 尝试写入默认配置
    return DEFAULT_PROCESSOR_CONFIG, DEFAULT_FEISHU_CONFIG, DEFAULT_POST_PROCESSING_COLS


def save_config(config_data):
    """将配置数据写入 config.json 文件。"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        print(f"配置已成功保存到 {CONFIG_FILE}。")
        return True
    except IOError as e:
        print(f"保存配置文件 {CONFIG_FILE} 时出错: {e}")
        return False


# 应用启动时加载配置
CURRENT_LLM_CONFIG, CURRENT_FEISHU_CONFIG, CURRENT_POST_PROCESSING_CONFIG = (
    load_config()
)

# 启动时检查 API Key (可以移到 load_config 内部或保持在这里)
# (保持原样)

# *** 新增：确保加载的配置也包含"来源" ***
if "来源" not in CURRENT_LLM_CONFIG.get("TARGET_COLUMNS", []):
    print("   * 配置加载后：将缺失的 '来源' 添加到 TARGET_COLUMNS。")
    CURRENT_LLM_CONFIG.setdefault("TARGET_COLUMNS", []).append("来源")


# 定义根路由，用于显示主页面 (index.html)
@app.route("/")
def index():
    """
    渲染应用程序主页面，提供文件上传和配置设置界面。

    显示Web界面，允许用户上传Excel/CSV文件进行处理，
    配置LLM、飞书和后处理参数，并查看任务处理进度。
    从服务器加载默认配置参数并传递给模板用于初始化界面表单。

    Returns:
        HTML: 渲染后的index.html页面，包含所有配置参数
    """
    # print(f"Rendering index with post_processing_defaults: {CURRENT_POST_PROCESSING_CONFIG}") # Debug
    return render_template(
        "index.html",
        default_llm_config=CURRENT_LLM_CONFIG,
        default_feishu_config=CURRENT_FEISHU_CONFIG,
        default_post_processing_cols=CURRENT_POST_PROCESSING_CONFIG,  # 将加载的默认后处理配置传递给模板
    )


# 定义文件上传路由，接收 POST 请求
@app.route("/upload", methods=["POST"])
def upload_files():
    """
    处理用户上传的Excel/CSV文件并启动后台处理任务。

    接收前端上传的文件并保存到任务专属目录，解析表单中的LLM、飞书和
    后处理配置参数，创建唯一任务ID，启动后台线程执行数据处理流程。
    处理流程包括LLM处理、飞书数据获取、数据合并和后处理检查。

    Request:
        - files[]: 上传的Excel/CSV文件列表
        - config_target_columns: 目标列名称列表(JSON)
        - config_api_key: DeepSeek API密钥
        - config_batch_size: 批处理大小
        - config_max_tokens: 最大Token数
        - config_api_timeout: API超时时间
        - feishu_*: 多个飞书API配置参数
        - post_process_config: 后处理配置(JSON)

    Returns:
        JSON: 包含task_id的响应，用于前端查询任务进度
    """
    # 基础验证：检查请求中是否包含文件部分
    if "files[]" not in request.files:
        return jsonify({"error": "请求中缺少 files[] 文件部分"}), 400
    # 获取所有上传的文件列表
    files = request.files.getlist("files[]")
    # 验证是否选择了文件
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "未选择任何文件"}), 400

    # 为本次任务生成唯一的 ID
    task_id = str(uuid.uuid4())
    # 根据 task_id 创建独立的上传和输出子目录
    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], task_id)
    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_file_paths = []
    try:
        # 遍历上传的文件列表
        for file in files:
            # 确保文件存在且有文件名
            if file and file.filename:
                # 安全考虑：可以使用 secure_filename 清理文件名
                # from werkzeug.utils import secure_filename
                # filename = secure_filename(file.filename)
                filename = file.filename
                # 构造文件保存路径
                filepath = os.path.join(upload_dir, filename)
                # 保存文件
                file.save(filepath)
                # 将保存后的文件路径添加到列表
                input_file_paths.append(filepath)
            else:
                print(f"跳过无效的文件条目: {file}")

        # 如果没有成功保存任何文件
        if not input_file_paths:
            # (可选) 清理已创建的空目录
            try:
                if os.path.exists(upload_dir):
                    os.rmdir(upload_dir)
                if os.path.exists(output_dir):
                    os.rmdir(output_dir)
            except OSError:
                pass  # 忽略清理错误
            return jsonify({"error": "没有有效的上传文件"}), 400

        # --- 从前端请求中获取配置参数 ---
        try:
            # LLM Processor Config
            target_columns_json = request.form.get(
                "config_target_columns", "[]"
            )  # Use new name
            target_columns = json.loads(target_columns_json)
            if not isinstance(target_columns, list) or not all(
                isinstance(s, str) for s in target_columns
            ):
                target_columns = CURRENT_LLM_CONFIG[
                    "TARGET_COLUMNS"
                ]  # Use updated default
                print(
                    f"警告: 前端发送的 target_columns 格式无效，使用服务器默认值: {target_columns}"
                )

            # *** 新增：强制确保"来源"列包含在任务配置中 ***
            source_column_name = "来源"
            if source_column_name not in target_columns:
                print(f"   * 任务配置：强制添加 '{source_column_name}' 到目标列。")
                target_columns.append(source_column_name)
            # *** 新增结束 ***

            # 如果前端未提供 API Key，则使用默认配置中的 Key
            api_key = request.form.get(
                "config_api_key", CURRENT_LLM_CONFIG["DEEPSEEK_API_KEY"]
            ).strip()  # Use new name
            if not api_key:  # 如果用户清空了输入框，仍然使用加载的默认值
                api_key = CURRENT_LLM_CONFIG["DEEPSEEK_API_KEY"]
                print(f"警告: 前端未提供有效 API Key，使用服务器默认 Key (可能无效)")

            # 对于数字类型，增加 try-except 保证健壮性
            try:
                batch_size = int(request.form.get("config_batch_size"))
            except (ValueError, TypeError):
                batch_size = CURRENT_LLM_CONFIG["BATCH_SIZE"]

            try:
                max_tokens = int(request.form.get("config_max_tokens"))
            except (ValueError, TypeError):
                max_tokens = CURRENT_LLM_CONFIG["MAX_COMPLETION_TOKENS"]

            try:
                api_timeout = int(request.form.get("config_api_timeout"))
            except (ValueError, TypeError):
                api_timeout = CURRENT_LLM_CONFIG["API_TIMEOUT"]

            # 可选: 添加更严格的参数验证 (例如 batch_size > 0)
            if batch_size <= 0:
                batch_size = CURRENT_LLM_CONFIG["BATCH_SIZE"]
            if max_tokens <= 0:
                max_tokens = CURRENT_LLM_CONFIG["MAX_COMPLETION_TOKENS"]
            if api_timeout <= 5:
                api_timeout = CURRENT_LLM_CONFIG["API_TIMEOUT"]  # 使用加载的默认超时

            # 创建本次任务的动态配置字典
            dynamic_config = {
                "TARGET_COLUMNS": target_columns,  # Now guaranteed to include "来源"
                "DEEPSEEK_API_KEY": api_key,
                "DEEPSEEK_API_ENDPOINT": CURRENT_LLM_CONFIG[
                    "DEEPSEEK_API_ENDPOINT"
                ],  # API端点通常不变
                "BATCH_SIZE": batch_size,
                "MAX_COMPLETION_TOKENS": max_tokens,
                "API_TIMEOUT": api_timeout,  # 新增超时配置
            }
            print(
                f"任务 {task_id} 使用配置: TargetCols={target_columns}, BatchSize={batch_size}, MaxTokens={max_tokens}, Timeout={api_timeout}"
            )

            # Feishu Config (使用加载的默认值)
            feishu_app_id = request.form.get(
                "feishu_app_id", CURRENT_FEISHU_CONFIG["APP_ID"]
            ).strip()
            feishu_app_secret = request.form.get(
                "feishu_app_secret", CURRENT_FEISHU_CONFIG["APP_SECRET"]
            ).strip()
            feishu_app_token = request.form.get(
                "feishu_app_token", CURRENT_FEISHU_CONFIG["APP_TOKEN"]
            ).strip()
            feishu_table_ids_json = request.form.get("feishu_table_ids", "[]")
            feishu_table_ids = json.loads(feishu_table_ids_json)
            if not isinstance(feishu_table_ids, list) or not all(
                isinstance(s, str) for s in feishu_table_ids
            ):
                feishu_table_ids = CURRENT_FEISHU_CONFIG["TABLE_IDS"]
                print(f"警告: 前端发送的 feishu_table_ids 格式无效，使用加载的默认值。")
            # 不再从表单读取这些列名，直接使用加载的默认值
            feishu_company_col = CURRENT_FEISHU_CONFIG.get(
                "COMPANY_NAME_COLUMN", "企业名称"
            )
            feishu_phone_col = CURRENT_FEISHU_CONFIG.get("PHONE_NUMBER_COLUMN", "电话")
            feishu_remark_col = CURRENT_FEISHU_CONFIG.get("REMARK_COLUMN_NAME", "备注")

            # Post Processing Choices (now a config object)
            post_process_config_json = request.form.get(
                "post_process_config", "{}"
            )  # Get the JSON string
            try:
                post_process_config = json.loads(post_process_config_json)  # Parse JSON
                if not isinstance(post_process_config, dict):
                    print(
                        f"警告: 前端发送的 post_process_config 不是有效的 JSON 对象，使用空配置。"
                    )
                    post_process_config = {}
            except json.JSONDecodeError:
                print(f"警告: 前端发送的 post_process_config 格式无效，使用空配置。")
                post_process_config = {}

            # 合并所有配置到一个字典
            task_config = {
                "llm_config": dynamic_config,  # LLM 相关配置
                "feishu_config": {
                    "APP_ID": feishu_app_id,
                    "APP_SECRET": feishu_app_secret,
                    "APP_TOKEN": feishu_app_token,
                    "TABLE_IDS": feishu_table_ids,
                    "ADD_TARGET_TABLE_IDS": CURRENT_FEISHU_CONFIG[
                        "ADD_TARGET_TABLE_IDS"
                    ],  # Add the new list
                    "COMPANY_NAME_COLUMN": feishu_company_col,
                    "PHONE_NUMBER_COLUMN": feishu_phone_col,
                    "REMARK_COLUMN_NAME": feishu_remark_col,
                    "RELATED_COMPANY_COLUMN_NAME": CURRENT_FEISHU_CONFIG[
                        "RELATED_COMPANY_COLUMN_NAME"
                    ],  # Ensure this is included
                },
                "post_processing_config": post_process_config,  # Store the entire post-processing config object
            }
            # 打印部分配置信息用于调试
            print(
                f"任务 {task_id} 使用配置: LLM Batch={dynamic_config.get('BATCH_SIZE')}, "
                f"Feishu Tables={len(feishu_table_ids)}, Add Targets={len(CURRENT_FEISHU_CONFIG['ADD_TARGET_TABLE_IDS'])}, "
                f"Feishu Remark Col={feishu_remark_col}, "
                f"Post Config={post_process_config}"  # Updated log
            )

        except (ValueError, TypeError, json.JSONDecodeError) as config_err:
            print(f"解析前端配置参数时出错: {config_err}，将使用默认配置。")
            # 如果解析出错，回退到默认配置 (需要合并默认的 LLM 和 Feishu 配置)
            task_config = {
                "llm_config": CURRENT_LLM_CONFIG.copy(),
                "feishu_config": CURRENT_FEISHU_CONFIG.copy(),
                "post_processing_config": {},  # Default empty post-processing config on error
            }

        # 定义最终输出的 Excel 文件名
        output_filename = f"consolidated_{task_id}.xlsx"
        # 构造完整的输出文件路径
        output_filepath = os.path.join(output_dir, output_filename)

        # 初始化任务状态信息
        tasks[task_id] = {
            "status": "Queued",  # 状态：已入队
            "progress": 0,  # 进度百分比
            "total_files": len(input_file_paths),  # 总文件数
            "files_processed": 0,  # 已处理文件数
            "result_file": None,  # 结果文件名 (处理完成后设置)
            "error": None,  # 错误信息 (如果发生错误)
        }

        print(f"为 {len(input_file_paths)} 个文件启动后台任务 {task_id}")
        # 创建并启动一个后台线程来执行实际的数据处理
        thread = threading.Thread(
            target=run_processing,  # 线程执行的目标函数
            args=(
                task_id,
                input_file_paths,
                output_filepath,
                task_config,
            ),  # 传递合并后的总配置
        )
        thread.start()

        # 返回任务 ID 给前端，以便后续查询进度
        return jsonify({"task_id": task_id})

    except Exception as e:
        print(f"保存任务 {task_id} 的上传文件时出错: {e}")
        # 保存文件出错时，尝试清理已创建的目录
        try:
            if os.path.exists(upload_dir):
                os.rmdir(
                    upload_dir
                )  # 如果目录非空，rmdir 会失败，更健壮的做法是用 shutil.rmtree
            if os.path.exists(output_dir):
                os.rmdir(output_dir)  # 同上
        except OSError as rm_err:
            print(f"上传文件保存失败后进行清理时出错: {rm_err}")
        return jsonify({"error": "保存上传文件失败"}), 500


# 定义一个回调函数，用于更新任务进度
def update_task_progress(
    task_id, status_msg, progress_pct, files_processed, total_files
):
    """由 processor 模块调用的回调函数，用于更新全局 tasks 字典中的任务状态。"""
    if task_id in tasks:
        tasks[task_id]["status"] = status_msg
        # 确保进度值在 0 到 100 之间
        tasks[task_id]["progress"] = max(0, min(100, int(progress_pct)))
        tasks[task_id]["files_processed"] = files_processed
        tasks[task_id]["total_files"] = total_files  # 顺便更新总文件数，以防万一
        # 在服务器控制台也打印进度信息，方便调试
        print(
            f"任务 {task_id} 进度: {status_msg} - {progress_pct}% ({files_processed}/{total_files} 文件)"
        )
    else:
        # 如果任务 ID 未知 (理论上不应发生)，打印警告
        print(f"警告: 尝试为未知的任务 ID 更新进度: {task_id}")


# 这个函数在后台线程中运行，负责调用核心处理逻辑
def run_processing(task_id, input_files, output_file, config):
    """在后台线程中运行的包装函数，调用 processor 的主处理函数并更新任务状态。"""
    try:
        # 定义一个内部回调函数，传递给 processor
        # 这个内部函数会调用外部的 update_task_progress 来更新全局状态
        def progress_callback(status_msg, progress_pct, files_processed, total_files):
            update_task_progress(
                task_id, status_msg, progress_pct, files_processed, total_files
            )

        print(f"任务 {task_id}: 开始执行处理流程...")
        # --- Stage 1: LLM Processing ---
        update_task_progress(
            task_id, "阶段1: 开始 LLM 处理上传的文件...", 5, 0, len(input_files)
        )
        # 调用 processor 模块的核心处理函数
        result_path = processor.process_files_and_consolidate(
            input_files=input_files,  # 输入文件路径列表
            output_file_path=output_file,  # 输出文件路径
            config=config["llm_config"],  # 传递 LLM 相关配置
            update_progress_callback=progress_callback,  # 进度回调函数
        )

        # Stage 1 完成，获取处理后的 DataFrame (如果需要，processor 可以返回 df)
        # 这里我们假设结果只保存在 result_path 文件中
        update_task_progress(
            task_id, "阶段1: LLM 处理完成", 30, len(input_files), len(input_files)
        )
        print(f"任务 {task_id}: LLM 处理阶段完成，结果保存在: {result_path}")

        # --- Stage 2: Fetch Feishu Data (此处添加代码) ---
        update_task_progress(
            task_id,
            "阶段2: 开始获取飞书数据...",
            35,
            len(input_files),
            len(input_files),
        )
        print(f"任务 {task_id}: 开始获取飞书数据... 配置: {config['feishu_config']}")

        df_feishu = pd.DataFrame()  # 初始化空的 DataFrame
        # 只有在提供了必要的飞书配置时才尝试获取
        if (
            config.get("feishu_config")
            and config["feishu_config"].get("APP_ID")
            and config["feishu_config"].get("APP_SECRET")
            and config["feishu_config"].get("APP_TOKEN")
            and config["feishu_config"].get("TABLE_IDS")
        ):
            try:
                df_feishu = feishu_utils.fetch_and_prepare_feishu_data(
                    config["feishu_config"]
                )
                print(f"任务 {task_id}: 飞书数据获取完成，共 {len(df_feishu)} 条记录。")
                update_task_progress(
                    task_id,
                    f"阶段2: 飞书数据获取完成 ({len(df_feishu)}条)",
                    50,
                    len(input_files),
                    len(input_files),
                )
            except Exception as feishu_err:
                print(f"任务 {task_id}: 获取飞书数据时发生错误: {feishu_err}")
                # 更新状态为错误，并可能需要停止后续流程或跳过合并
                update_task_progress(
                    task_id,
                    f"阶段2: 飞书数据获取失败 - {feishu_err}",
                    50,
                    len(input_files),
                    len(input_files),
                )
                # 决定是否继续，这里暂时允许继续，但 df_feishu 会是空的
                pass  # 或者 raise feishu_err 停止整个流程
        else:
            print(f"任务 {task_id}: 未提供完整的飞书配置，跳过飞书数据获取。")
            update_task_progress(
                task_id,
                "阶段2: 跳过飞书数据获取 (配置不完整)",
                50,
                len(input_files),
                len(input_files),
            )

        # --- Stage 3: Merge Data (此处添加代码) ---
        update_task_progress(
            task_id, "阶段3: 开始合并数据...", 55, len(input_files), len(input_files)
        )

        merged_df = pd.DataFrame()  # 初始化空的 DataFrame
        merged_output_filename = f"merged_{task_id}.xlsx"
        merged_output_path = os.path.join(
            os.path.dirname(output_file), merged_output_filename
        )

        try:
            print(f"任务 {task_id}: 开始合并数据...")
            # 读取 Stage 1 (LLM 处理) 的结果 Excel 文件
            if os.path.exists(result_path):
                df_processed = pd.read_excel(result_path)
                print(f"   读取 LLM 处理结果成功，共 {len(df_processed)} 行。")
            else:
                print(
                    f"   警告: 未找到 LLM 处理结果文件 {result_path}。将只使用飞书数据 (如果存在)。"
                )
                df_processed = pd.DataFrame()

            # 准备合并
            target_columns = config.get("llm_config", {}).get(
                "TARGET_COLUMNS", []
            )  # Guaranteed to have "来源"
            remark_column = config.get("feishu_config", {}).get(
                "REMARK_COLUMN_NAME", "备注"
            )
            related_company_col = config.get("feishu_config", {}).get(
                "RELATED_COMPANY_COLUMN_NAME",
                "关联公司名称(LLM)",  # 从配置获取或使用默认
            )
            local_id_col = "local_row_id"
            source_column_name = "来源"  # Define for clarity

            # 最终需要的列 - re-derive ensuring all required columns are present
            # Start with target_columns (which includes '来源'), add others, then deduplicate
            required_cols = [
                "record_id",
                remark_column,
                related_company_col,
                local_id_col,
            ]
            final_columns_set = set(target_columns + required_cols)
            # Optional: Define a specific order if needed, otherwise set conversion is fine
            final_columns = list(
                dict.fromkeys(target_columns + required_cols)
            )  # Keep order from target + append others
            print(f"   最终合并列 (含来源): {final_columns}")

            # 标准化 df_processed 列 (should already have '来源' from processor.py)
            print("   标准化 LLM 处理结果列 (验证来源列)...")
            if not df_processed.empty:
                if source_column_name not in df_processed.columns:
                    print(
                        f"   ⚠️ 警告: LLM 处理结果中意外缺失 '{source_column_name}'。添加空列。"
                    )
                    df_processed[source_column_name] = ""
                # Ensure other required columns exist
                for col in final_columns:
                    if col not in df_processed.columns:
                        default_value = (
                            None if col == "record_id" or col == local_id_col else ""
                        )
                        df_processed[col] = default_value
                df_processed = df_processed[final_columns]  # Reorder/select columns
            else:
                df_processed = pd.DataFrame(columns=final_columns)

            # 标准化 df_feishu 列 (如果存在)
            if not df_feishu.empty:
                print("   开始标准化飞书数据列 (添加来源列)...")
                # *** 新增：确保飞书数据有"来源"列，并设置为空字符串 ***
                if source_column_name not in df_feishu.columns:
                    df_feishu[source_column_name] = ""
                # *** 新增结束 ***
                # Ensure other required columns exist in Feishu data
                for col in final_columns:
                    if col not in df_feishu.columns:
                        default_value = (
                            None if col == "record_id" or col == local_id_col else ""
                        )
                        df_feishu[col] = default_value
                # Select and reorder columns from Feishu data
                cols_to_keep_from_feishu = [
                    col for col in final_columns if col in df_feishu.columns
                ]
                if not cols_to_keep_from_feishu:
                    df_feishu = pd.DataFrame(columns=final_columns)
                else:
                    df_feishu = df_feishu[final_columns]  # Reorder/select final columns
                print(f"   飞书数据标准化完成，列: {df_feishu.columns.tolist()}")
            else:
                print("   无飞书数据需要标准化。")

            # 合并数据
            if not df_processed.empty or not df_feishu.empty:
                merged_df = pd.concat([df_processed, df_feishu], ignore_index=True)
                print(f"   数据合并完成，总计 {len(merged_df)} 行。")
                # 保存合并后的文件 (用于后续处理)
                merged_df.to_excel(merged_output_path, index=False, engine="openpyxl")
                print(f"   合并后的数据已保存到: {merged_output_path}")
                # 更新任务状态，提供合并文件下载链接
                tasks[task_id]["merged_file"] = merged_output_filename
                update_task_progress(
                    task_id,
                    f"阶段3: 数据合并完成 ({len(merged_df)}条)",
                    60,
                    len(input_files),
                    len(input_files),
                )
            else:
                print("   没有可合并的数据。跳过合并阶段。")
                merged_df = pd.DataFrame(
                    columns=final_columns
                )  # 确保 merged_df 有正确的列结构
                update_task_progress(
                    task_id,
                    "阶段3: 无数据可合并",
                    60,
                    len(input_files),
                    len(input_files),
                )

        except Exception as merge_err:
            print(f"任务 {task_id}: 合并数据时发生错误: {merge_err}")
            traceback.print_exc()
            update_task_progress(
                task_id,
                f"阶段3: 数据合并失败 - {merge_err}",
                60,
                len(input_files),
                len(input_files),
            )
            merged_df = pd.DataFrame(columns=final_columns)  # 出错时也确保有列结构
            pass  # 或者 raise merge_err

        # --- Stage 4: Post Processing (此处添加代码) ---
        update_task_progress(
            task_id,
            "阶段4: 开始执行后处理检查...",
            65,
            len(input_files),
            len(input_files),
        )
        # 移除对 merged_df 的默认赋值，因为下面会处理
        # final_df = merged_df
        final_output_filename = f"final_{task_id}.xlsx"
        final_output_path = os.path.join(
            os.path.dirname(output_file), final_output_filename
        )
        final_df = pd.DataFrame()  # 初始化 final_df

        # Retrieve post-processing config for this task
        task_post_config = config.get("post_processing_config", {})
        # Get the actual choices (keys of the config object)
        post_choices = list(task_post_config.keys())
        print(
            f"任务 {task_id}: 开始执行后处理... 选项: {post_choices}, 配置: {task_post_config}"
        )

        if not merged_df.empty and post_choices:
            try:
                print(
                    f"   输入后处理的数据有 {len(merged_df)} 行，列: {merged_df.columns.tolist()}"
                )
                # 调用后处理函数
                final_df = postprocessor.apply_post_processing(
                    merged_df.copy(), config
                )  # 传递副本以防万一
                print(
                    f"   后处理返回的数据有 {len(final_df)} 行，列: {final_df.columns.tolist()}"
                )
                # 使用多Sheet页保存结果
                postprocessor.create_multi_sheet_excel(
                    final_df, final_output_path, config
                )
                print(
                    f"任务 {task_id}: 后处理完成，多Sheet页结果保存在: {final_output_path}"
                )
                update_task_progress(
                    task_id, "阶段4: 后处理完成", 95, len(input_files), len(input_files)
                )
            except Exception as post_err:
                print(f"任务 {task_id}: 后处理过程中发生错误: {post_err}")
                traceback.print_exc()
                # 即使后处理失败，也尝试保存合并后的数据作为最终结果
                print("   ⚠️ 后处理失败，尝试保存合并后的数据作为最终结果。")
                final_df = merged_df  # 数据回退到合并后的数据
                try:
                    final_output_path = os.path.join(
                        os.path.dirname(output_file), final_output_filename
                    )
                    print(f"   最终保存路径: {final_output_path}")
                    # 使用多Sheet页Excel保存最终结果文件
                    postprocessor.create_multi_sheet_excel(
                        final_df, final_output_path, config
                    )
                    print(f"   最终文件 {final_output_filename} 保存成功。")
                except Exception as save_err:
                    print(
                        f"   ❌ 尝试保存合并数据时也失败了: {save_err}. 将使用空结果。"
                    )
                    final_df = pd.DataFrame()  # 保存也失败，返回空
                    final_output_filename = None  # 没有结果文件
                update_task_progress(
                    task_id,
                    f"阶段4: 后处理失败 - {post_err}",
                    95,
                    len(input_files),
                    len(input_files),
                )
        elif merged_df.empty:
            print(f"任务 {task_id}: 合并数据为空，跳过后处理。最终结果将为空。")
            # final_output_filename = None # 保持为 None 或创建一个空文件？创建一个空文件吧
            final_df = pd.DataFrame(columns=final_columns)  # 保持列结构
            try:
                # 使用多Sheet页保存结果
                postprocessor.create_multi_sheet_excel(
                    final_df, final_output_path, config
                )
                print(f"   创建了空的最终文件: {final_output_path}")
            except Exception as save_empty_err:
                print(f"   ❌ 创建空最终文件时失败: {save_empty_err}")
                final_output_filename = None  # 保存失败，则无文件
            update_task_progress(
                task_id,
                "阶段4: 跳过后处理 (无数据)",
                95,
                len(input_files),
                len(input_files),
            )
        else:  # merged_df 不为空，但 post_choices 为空
            print(
                f"任务 {task_id}: 未选择任何后处理选项，将合并结果直接保存为最终结果。"
            )
            final_df = merged_df  # 最终数据就是合并数据
            try:
                # 使用多Sheet页保存结果
                postprocessor.create_multi_sheet_excel(
                    final_df, final_output_path, config
                )
                final_output_filename = final_output_filename  # 保持最终文件名
                print(f"   合并结果已直接保存为多Sheet页最终文件: {final_output_path}")
            except Exception as save_merged_err:
                print(f"   ❌ 保存合并结果时出错: {save_merged_err}，最终结果将为空。")
                final_df = pd.DataFrame()  # 保存失败，返回空
                final_output_filename = None  # 没有结果文件
            update_task_progress(
                task_id,
                "阶段4: 跳过后处理 (未选择选项)",
                95,
                len(input_files),
                len(input_files),
            )

        # --- Stage 5: Finalize ---
        # *** FIX: Ensure all rows have a unique local_row_id before saving ***
        final_local_id_col = "local_row_id"  # Make sure variable name is defined
        if final_local_id_col in final_df.columns:
            missing_ids_mask = final_df[final_local_id_col].isna() | (
                final_df[final_local_id_col] == ""
            )
            num_missing = missing_ids_mask.sum()
            if num_missing > 0:
                print(
                    f"   检测到 {num_missing} 行缺少 local_row_id (可能来自飞书)，正在生成唯一 ID..."
                )
                # Generate unique IDs only for rows that are missing them
                final_df.loc[missing_ids_mask, final_local_id_col] = [
                    f"generated_uuid_{uuid.uuid4()}" for _ in range(num_missing)
                ]
                # Verify (optional)
                # print(f"   生成后检查，缺失数: {final_df[final_local_id_col].isna().sum()}")
        else:
            print(
                f"   警告: 最终 DataFrame 中未找到列 '{final_local_id_col}'，无法填充缺失的 ID。"
            )

        # Save the final dataframe (now with guaranteed unique local_row_ids)
        try:
            # Ensure the output path uses the correct filename variable
            if final_output_filename:
                final_output_path = os.path.join(
                    os.path.dirname(output_file), final_output_filename
                )
                print(f"   最终保存路径: {final_output_path}")
                # 使用多Sheet页Excel保存最终结果文件
                postprocessor.create_multi_sheet_excel(
                    final_df, final_output_path, config
                )
                print(f"   最终文件 {final_output_filename} 保存成功。")
            else:
                print("   ❌ 错误: 最终文件名未确定，无法保存最终结果。")
                # Handle error - perhaps raise or set task status to error
                raise ValueError("最终文件名丢失，无法保存结果")

        except Exception as final_save_err:
            print(
                f"   ❌ 保存最终文件 {final_output_filename} 时出错: {final_save_err}"
            )
            traceback.print_exc()
            # Even if save fails here, we might want to report completion but without a result file?
            final_output_filename = None  # Indicate file wasn't saved
            # Fall through to task completion update, but with no result file

        # Update task state after attempting final save
        tasks[task_id]["status"] = "Completed" if final_output_filename else "Error"
        tasks[task_id]["progress"] = 100
        tasks[task_id][
            "result_file"
        ] = final_output_filename  # Use the potentially updated filename
        tasks[task_id]["config"] = config  # 保存任务配置，供后续差异比较使用

    except Exception as e:
        # 如果处理过程中发生任何异常
        error_message = f"处理过程中出错: {type(e).__name__} - {e}"
        print(f"任务 {task_id}: {error_message}")
        # 在服务器控制台打印详细的错误堆栈信息，方便调试
        traceback.print_exc()
        # 更新任务状态为 "Error"
        if task_id in tasks:
            tasks[task_id]["status"] = "Error"
            tasks[task_id]["error"] = error_message
            # 即使出错，也将进度设为 100，告知前端轮询可以停止
            tasks[task_id]["progress"] = 100
    finally:
        # 无论成功还是失败，线程结束时都会执行
        # 可选：在这里添加清理上传文件的逻辑
        # 目前为了方便调试，暂时保留上传的文件
        print(f"任务 {task_id}: 后台处理线程结束。")
        # 清理示例 (如果需要，取消注释并调整):
        # upload_dir = os.path.dirname(input_files[0]) # 获取该任务的上传目录
        # print(f"任务 {task_id}: 尝试清理上传目录 {upload_dir}")
        # import shutil # 需要导入 shutil 模块来删除非空目录
        # try:
        #     shutil.rmtree(upload_dir) # 删除整个目录及其内容
        #     print(f"任务 {task_id}: 成功删除上传目录 {upload_dir}")
        # except Exception as cleanup_err:
        #     print(f"任务 {task_id}: 清理上传目录 {upload_dir} 时出错 - {cleanup_err}")


# 定义进度查询路由
@app.route("/progress/<task_id>")
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
    task = tasks.get(task_id)
    if not task:
        # 如果任务 ID 不存在 (可能已完成很久被清理，或从未存在)
        # 返回 404 错误，并提供一个表示任务未知或结束的状态
        return (
            jsonify(
                {
                    "status": "未知或已过期的任务",
                    "progress": 100,  # 让前端停止轮询
                    "error": "任务 ID 未找到。",
                }
            ),
            404,
        )
    # 返回当前任务的状态信息
    return jsonify(task)


# 定义文件下载路由
@app.route("/download/<task_id>/<filename>")
def download_file(task_id, filename):
    """
    下载指定任务的处理结果文件。

    根据任务ID和文件名从输出目录提供文件下载。包含安全检查防止路径遍历攻击，
    确保文件存在且属于指定的任务。各种错误情况会返回相应的HTTP错误码和信息。

    Path Params:
        - task_id: 任务唯一标识符UUID
        - filename: 要下载的文件名，如"final_xxx.xlsx"

    Returns:
        File: 处理结果文件的下载响应
        或 JSON错误信息: 文件不存在/目录不存在/文件名无效
    """
    # 构造该任务的输出目录路径
    directory = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
    print(f"尝试下载: {filename} 从目录 {directory}")

    # 检查输出目录是否存在
    if not os.path.exists(directory):
        print(f"下载失败: 目录未找到 - {directory}")
        return jsonify({"error": "任务输出目录未找到"}), 404

    # 安全性检查: 防止路径遍历攻击 (尽管这里路径是内部构造的，但加上更好)
    if ".." in filename or filename.startswith("/"):
        print(f"下载失败: 文件名无效 - {filename}")
        return jsonify({"error": "文件名无效"}), 400

    # 构造完整的文件路径
    file_path = os.path.join(directory, filename)
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"下载失败: 文件未找到 - {file_path}")
        return jsonify({"error": "结果文件未找到"}), 404

    try:
        # 使用 Flask 的 send_from_directory 函数安全地发送文件
        # as_attachment=True 会让浏览器弹出下载对话框
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        # 处理发送文件过程中的其他潜在错误
        print(f"为任务 {task_id} 发送文件 {filename} 时出错: {e}")
        return jsonify({"error": "服务器发送文件时出错"}), 500


# 新增：保存配置的路由
@app.route("/save_config", methods=["POST"])
def save_config_route():
    """
    保存用户定义的配置为系统默认值。

    接收JSON格式的配置数据，包含LLM、飞书和后处理配置，
    更新内存中的配置变量并写入config.json文件，
    使配置在下次应用启动时自动生效作为默认值。

    Request Body:
        JSON对象，必须包含三个键:
        - llm_config: LLM处理相关配置(API密钥、批量大小等)
        - feishu_config: 飞书API配置(ID、密钥、表IDs等)
        - post_processing_config: 数据后处理配置

    Returns:
        JSON: 表示保存成功或失败的消息
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON"}), 400

    new_config_data = request.get_json()

    # 基础验证传入的数据结构 (可以根据需要做得更细致)
    if (
        not isinstance(new_config_data, dict)
        or "llm_config" not in new_config_data
        or "feishu_config" not in new_config_data
        or "post_processing_config" not in new_config_data  # 确保新配置也包含后处理部分
    ):
        return (
            jsonify({"success": False, "error": "Invalid config data structure"}),
            400,
        )

    # 更新当前的配置变量 (这样下次渲染页面时就是新的默认值了)
    # 注意：这里直接修改全局变量，简单但不适合复杂应用
    global CURRENT_LLM_CONFIG, CURRENT_FEISHU_CONFIG, CURRENT_POST_PROCESSING_CONFIG
    # 更新LLM和飞书配置
    CURRENT_LLM_CONFIG.update(new_config_data.get("llm_config", {}))
    CURRENT_FEISHU_CONFIG.update(new_config_data.get("feishu_config", {}))
    # 特别注意：保存的是用户界面上当前的后处理设置作为 *新的默认值*
    CURRENT_POST_PROCESSING_CONFIG = new_config_data.get("post_processing_config", {})
    print(
        f"Updated default post-processing config to: {CURRENT_POST_PROCESSING_CONFIG}"
    )  # Debug

    # 将合并后的完整配置写入文件
    full_config_to_save = {
        "llm_config": CURRENT_LLM_CONFIG,
        "feishu_config": CURRENT_FEISHU_CONFIG,
        "post_processing_config": CURRENT_POST_PROCESSING_CONFIG,  # 保存更新后的后处理默认值
    }

    if save_config(full_config_to_save):
        return jsonify({"success": True, "message": "配置已成功保存为默认值！"})
    else:
        return (
            jsonify({"success": False, "error": "保存配置文件时发生服务器错误。"}),
            500,
        )


# 新增：上传用户修改后的文件的路由
@app.route("/upload_edited/<task_id>", methods=["POST"])
def upload_edited_file(task_id):
    """
    上传用户手动编辑后的Excel文件。

    在处理完成后，用户可能会下载结果进行人工修改和校正，
    此API用于接收修改后的文件，保存到任务目录并更新任务状态。
    上传后的文件将用于后续差异比较和飞书同步操作。

    Path Params:
        - task_id: 任务唯一标识符UUID

    Request:
        - edited_file: 用户编辑后的Excel文件

    Returns:
        JSON: 上传成功或失败信息，成功时包含编辑后的文件名
    """
    if task_id not in tasks:
        return jsonify({"success": False, "error": "任务 ID 不存在或已过期。"}), 404

    # 检查是否有文件上传
    if "edited_file" not in request.files:
        return (
            jsonify({"success": False, "error": "请求中缺少 'edited_file' 文件部分。"}),
            400,
        )

    file = request.files["edited_file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "未选择任何文件。"}), 400

    if file:  # 再次确认文件存在
        # 构建保存路径 (保存在该任务的输出目录下)
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
        if not os.path.exists(output_dir):
            # 如果输出目录意外丢失，尝试重新创建？或直接报错
            print(
                f"警告: 任务 {task_id} 的输出目录 {output_dir} 不存在，正在尝试创建。"
            )
            os.makedirs(output_dir, exist_ok=True)

        edited_filename = f"edited_{task_id}.xlsx"
        edited_filepath = os.path.join(output_dir, edited_filename)

        try:
            file.save(edited_filepath)
            print(f"任务 {task_id}: 已保存用户上传的修改后文件到 {edited_filepath}")

            # 更新任务状态，表示已收到修改后的文件
            tasks[task_id]["edited_file_uploaded"] = True
            tasks[task_id]["edited_file_path"] = edited_filepath
            # 可以在这里更新状态消息，提示用户可以进行差异检查
            # update_task_progress(task_id, "阶段4: 已上传调整后文件，可进行差异检查", 100, ...) # 进度可能不需要更新

            return jsonify(
                {
                    "success": True,
                    "message": "调整后的文件上传成功！",
                    "edited_filename": edited_filename,
                }
            )
        except Exception as e:
            print(f"任务 {task_id}: 保存上传的修改后文件时出错: {e}")
            traceback.print_exc()
            return (
                jsonify({"success": False, "error": "保存文件时发生服务器错误。"}),
                500,
            )

    return jsonify({"success": False, "error": "上传失败，未知错误。"}), 500


# 新增：执行差异比较的路由
@app.route("/check_diff/<task_id>", methods=["GET"])
def check_differences(task_id):
    """
    显示多Sheet页Excel文件中的Sheet页内容摘要。

    读取最终处理结果(多Sheet页Excel文件)，提取各Sheet页的摘要统计信息，
    包括每个Sheet页的行数和包含的数据概况。这些信息可以帮助用户了解
    数据处理后的分类结果，以便决定是否需要同步到飞书。

    Path Params:
        - task_id: 任务唯一标识符UUID

    Returns:
        JSON: Excel内容摘要对象，包含:
            - sheet_info: 各Sheet页的统计信息
            - total_rows: 所有Sheet页的总行数
            - columns: Excel文件包含的列名列表
    """
    if task_id not in tasks:
        return jsonify({"success": False, "error": "任务 ID 不存在或已过期。"}), 404

    task_info = tasks[task_id]

    # 检查必要的文件是否存在
    final_filename = task_info.get("result_file")  # 最终文件名

    if not final_filename:
        return jsonify({"success": False, "error": "找不到处理结果文件信息。"}), 404

    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
    final_filepath = os.path.join(output_dir, final_filename)

    if not os.path.exists(final_filepath):
        print(f"错误: 找不到任务 {task_id} 的结果文件: {final_filepath}")
        return jsonify({"success": False, "error": "找不到多Sheet页Excel文件。"}), 404

    print(f"任务 {task_id}: 开始读取多Sheet页Excel文件: '{final_filepath}'")

    try:
        # 读取Excel文件的各个Sheet
        sheet_info = {}
        total_rows = 0
        all_columns = set()
        record_id_col = "record_id"

        with pd.ExcelFile(final_filepath) as xls:
            sheets = xls.sheet_names
            print(f"  文件包含以下Sheet页: {sheets}")

            for sheet_name in sheets:
                # 读取每个Sheet
                df = pd.read_excel(xls, sheet_name=sheet_name)
                row_count = len(df)
                total_rows += row_count

                # 收集列名
                for col in df.columns:
                    all_columns.add(col)

                # 统计具体信息
                sheet_stats = {
                    "row_count": row_count,
                    "columns": list(df.columns),
                }

                # 对特殊Sheet页做额外统计
                if sheet_name in ["新增", "更新"]:
                    # 统计record_id情况
                    if record_id_col in df.columns:
                        empty_record_ids = df[
                            df[record_id_col].fillna("").astype(str).str.strip() == ""
                        ].shape[0]
                        valid_record_ids = row_count - empty_record_ids
                        sheet_stats["empty_record_ids"] = empty_record_ids
                        sheet_stats["valid_record_ids"] = valid_record_ids

                sheet_info[sheet_name] = sheet_stats

        # 构建响应
        result = {
            "success": True,
            "sheet_info": sheet_info,
            "total_rows": total_rows,
            "columns": list(all_columns),
        }

        print(
            f"任务 {task_id}: 多Sheet页Excel文件读取完成，共 {total_rows} 行数据，分布在 {len(sheets)} 个Sheet页"
        )
        return jsonify(result)

    except FileNotFoundError as fnf_err:
        print(f"任务 {task_id}: 读取Excel文件时出错: {fnf_err}")
        return (
            jsonify({"success": False, "error": f"读取文件失败: {fnf_err.filename}"}),
            500,
        )
    except Exception as e:
        print(f"任务 {task_id}: 读取Excel文件时发生错误: {e}")
        traceback.print_exc()
        return (
            jsonify({"success": False, "error": "读取Excel文件时发生服务器错误。"}),
            500,
        )


# 新增：将差异同步回飞书的路由
@app.route("/sync_to_feishu/<task_id>", methods=["POST"])
def sync_to_feishu(task_id):
    """
    将数据同步到飞书多维表格。

    基于多Sheet页Excel文件执行两种操作:
    1. 添加新记录 - 从"新增"Sheet页读取数据
    2. 更新记录 - 将"更新"Sheet页的数据与原始数据Sheet比较，只更新有差异的字段

    不再执行删除操作。

    执行前会验证配置、检查飞书表空间容量并格式化数据，
    同步结果会详细记录成功和失败的操作数量。

    Path Params:
        - task_id: 任务唯一标识符UUID

    Requires:
        - 已上传的编辑后文件(包含多Sheet页)
        - 有效的飞书API配置

    Returns:
        JSON: 同步操作结果，包含添加/更新/删除的记录数和错误信息
    """
    # --- 常量定义 ---
    FEISHU_ROW_LIMIT = 50000

    if task_id not in tasks:
        return jsonify({"success": False, "error": "任务 ID 不存在或已过期。"}), 404

    task_info = tasks[task_id]
    print(f"任务 {task_id}: 开始执行同步到飞书操作 (使用多Sheet页Excel)...")

    # --- 检查必需的信息和配置 --- #
    edited_filepath = task_info.get("edited_file_path")

    if not edited_filepath or not os.path.exists(edited_filepath):
        # Edited 文件必须存在
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
        edited_filename_fallback = f"edited_{task_id}.xlsx"
        edited_filepath_fallback = os.path.join(output_dir, edited_filename_fallback)
        if os.path.exists(edited_filepath_fallback):
            edited_filepath = edited_filepath_fallback
            task_info["edited_file_path"] = edited_filepath  # Update cache
        else:
            print(
                f"错误: 找不到任务 {task_id} 的多Sheet页文件: {edited_filepath_fallback}"
            )
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "找不到多Sheet页文件，无法执行同步操作。",
                    }
                ),
                404,
            )

    if (
        "config" not in task_info
        or "feishu_config" not in task_info["config"]
        or "llm_config" not in task_info["config"]
    ):
        return (
            jsonify(
                {"success": False, "error": "任务配置信息缺失，无法获取飞书或LLM配置。"}
            ),
            500,
        )

    # --- 获取配置 --- #
    feishu_config = task_info["config"]["feishu_config"]
    llm_config = task_info["config"]["llm_config"]
    app_id = feishu_config.get("APP_ID")
    app_secret = feishu_config.get("APP_SECRET")
    app_token = feishu_config.get("APP_TOKEN")
    primary_table_ids = feishu_config.get("TABLE_IDS")
    add_target_table_ids = feishu_config.get("ADD_TARGET_TABLE_IDS", [])
    target_columns = llm_config.get("TARGET_COLUMNS", [])

    # --- 配置验证 --- #
    if not app_id or not app_secret:
        msg = "飞书 App ID 或 App Secret 未配置"
    elif not app_token or not primary_table_ids:
        msg = "飞书 Base App Token 或主 Table IDs 配置缺失"
    else:
        msg = None
    if msg:
        print(f"❌ 配置错误: {msg}")
        flash(msg, "error")
        return jsonify({"success": False, "message": msg, "summary": {"errors": [msg]}})

    # --- 准备操作列表和结果字典 --- #
    feishu_id_col = "record_id"
    local_id_col = "local_row_id"
    records_to_update = []
    records_to_add = []
    results = {
        "updated": 0,
        "added": 0,
        "update_errors": 0,
        "add_errors": 0,
        "errors": [],
        "diff_details": [],  # 新增：记录差异详情
    }

    try:
        # --- 1. 从多Sheet页Excel文件读取各Sheet页数据 --- #
        print(f"  步骤 1: 从 '{edited_filepath}' 读取各Sheet页数据...")
        try:
            # 读取Excel文件的Sheet页
            with pd.ExcelFile(edited_filepath) as xls:
                sheets = xls.sheet_names
                print(f"    -> 文件包含以下Sheet页: {sheets}")

                # 确保所需Sheet都存在
                if "原始数据" not in sheets:
                    print(f"    -> 警告: 未找到'原始数据'Sheet页，无法进行差异比较")

                # 读取原始数据Sheet页
                df_original = pd.DataFrame()
                if "原始数据" in sheets:
                    df_original = pd.read_excel(xls, sheet_name="原始数据")
                    print(f"    -> 从'原始数据'Sheet读取了 {len(df_original)} 行数据")
                    # 确保record_id列存在并清理格式
                    if feishu_id_col in df_original.columns:
                        df_original[feishu_id_col] = (
                            df_original[feishu_id_col]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                        )
                        # 处理"none"值
                        df_original.loc[
                            df_original[feishu_id_col].str.lower() == "none",
                            feishu_id_col,
                        ] = ""
                    else:
                        print(
                            f"    -> 警告: '原始数据'Sheet中缺少'{feishu_id_col}'列，可能影响差异比较"
                        )

                # 读取"新增"Sheet页
                df_add = pd.DataFrame()
                if "新增" in sheets:
                    df_add = pd.read_excel(xls, sheet_name="新增")
                    print(f"    -> 从'新增'Sheet读取了 {len(df_add)} 行数据")
                else:
                    print(f"    -> 未找到'新增'Sheet，跳过新增操作")

                # 读取"更新"Sheet页
                df_update = pd.DataFrame()
                if "更新" in sheets:
                    df_update = pd.read_excel(xls, sheet_name="更新")
                    print(f"    -> 从'更新'Sheet读取了 {len(df_update)} 行数据")
                else:
                    print(f"    -> 未找到'更新'Sheet，跳过更新操作")

            # 准备新增记录
            if not df_add.empty:
                # 确保record_id列存在 (在新增数据中应该是空的)
                if feishu_id_col not in df_add.columns:
                    df_add[feishu_id_col] = ""
                else:
                    df_add[feishu_id_col] = (
                        df_add[feishu_id_col].fillna("").astype(str).str.strip()
                    )
                    # 处理"none"值
                    df_add.loc[
                        df_add[feishu_id_col].str.lower() == "none", feishu_id_col
                    ] = ""

                # 筛选确保只处理record_id为空的行
                df_add = df_add[df_add[feishu_id_col] == ""]
                print(f"    -> '新增'Sheet中有 {len(df_add)} 行有效数据(record_id为空)")

                # 准备新增数据 (与原代码相同)
                for _, row in df_add.iterrows():
                    add_payload = {"fields": {}}
                    for col in target_columns:
                        if col in row.index:
                            value = row[col]
                            # 忽略 None 值和空字符串
                            if (
                                pd.isna(value)
                                or value is None
                                or (isinstance(value, str) and not value.strip())
                            ):
                                continue

                            # 处理不同类型的值
                            if isinstance(value, (list, dict)):
                                value_str = json.dumps(value, ensure_ascii=False)
                            else:
                                value_str = str(value)

                            add_payload["fields"][col] = value_str

                    # 只有当fields非空时才添加到列表
                    if add_payload["fields"]:
                        records_to_add.append(add_payload)

                print(f"    -> 准备了 {len(records_to_add)} 条记录待新增。")

            # 准备更新记录 - 改为与原始数据对比
            if not df_update.empty:
                # 确保record_id列存在 (在更新数据中不应该为空)
                if feishu_id_col not in df_update.columns:
                    print(
                        f"    -> 警告: '更新'Sheet中缺少 '{feishu_id_col}' 列，无法执行更新操作"
                    )
                else:
                    # 清理record_id
                    df_update[feishu_id_col] = (
                        df_update[feishu_id_col].fillna("").astype(str).str.strip()
                    )
                    # 处理"none"值
                    df_update.loc[
                        df_update[feishu_id_col].str.lower() == "none", feishu_id_col
                    ] = ""

                    # 筛选有效记录
                    df_update = df_update[df_update[feishu_id_col] != ""]
                    print(
                        f"    -> '更新'Sheet中有 {len(df_update)} 行包含有效record_id"
                    )

                    # 修改: 准备更新数据 - 与原始数据比较
                    diff_count = 0
                    for _, update_row in df_update.iterrows():
                        record_id = update_row[feishu_id_col]

                        # 在原始数据中查找相同record_id的行
                        if (
                            not df_original.empty
                            and feishu_id_col in df_original.columns
                        ):
                            original_rows = df_original[
                                df_original[feishu_id_col] == record_id
                            ]

                            if original_rows.empty:
                                print(
                                    f"    -> 警告: 原始数据中找不到record_id '{record_id}'，该行将被完整更新"
                                )
                                # 准备完整更新
                                update_payload = {"record_id": record_id, "fields": {}}

                                for col in target_columns:
                                    if col in update_row.index and col != feishu_id_col:
                                        value = update_row[col]
                                        if pd.isna(value) or value is None:
                                            continue

                                        # 处理不同类型
                                        if isinstance(value, (list, dict)):
                                            processed_value = json.dumps(
                                                value, ensure_ascii=False
                                            )
                                        else:
                                            processed_value = str(value)

                                        update_payload["fields"][col] = processed_value

                                if update_payload["fields"]:
                                    records_to_update.append(update_payload)
                                    diff_count += 1
                            else:
                                # 存在原始行，进行字段级差异比较
                                original_row = original_rows.iloc[0]
                                update_payload = {"record_id": record_id, "fields": {}}
                                diff_found = False
                                diff_fields = []

                                for col in target_columns:
                                    if (
                                        col in update_row.index
                                        and col in original_row.index
                                        and col != feishu_id_col
                                    ):
                                        update_value = update_row[col]
                                        original_value = original_row[col]

                                        # 处理空值
                                        if pd.isna(update_value):
                                            update_value = ""
                                        if pd.isna(original_value):
                                            original_value = ""

                                        # 转换为字符串进行比较
                                        update_str = str(update_value).strip()
                                        original_str = str(original_value).strip()

                                        # 如果有差异，则添加到更新字段
                                        if update_str != original_str:
                                            diff_found = True
                                            diff_fields.append(
                                                f"{col}: '{original_str}' -> '{update_str}'"
                                            )

                                            if isinstance(update_value, (list, dict)):
                                                processed_value = json.dumps(
                                                    update_value, ensure_ascii=False
                                                )
                                            else:
                                                processed_value = update_str

                                            update_payload["fields"][
                                                col
                                            ] = processed_value

                                # 只有找到差异的记录才添加到更新列表
                                if diff_found and update_payload["fields"]:
                                    records_to_update.append(update_payload)
                                    diff_count += 1
                                    # 记录差异详情
                                    if diff_fields:
                                        results["diff_details"].append(
                                            {
                                                "record_id": record_id,
                                                "changes": diff_fields,
                                            }
                                        )
                        else:
                            # 原始数据为空或缺少record_id列，只能完整更新
                            update_payload = {"record_id": record_id, "fields": {}}

                            for col in target_columns:
                                if col in update_row.index and col != feishu_id_col:
                                    value = update_row[col]
                                    if pd.isna(value) or value is None:
                                        continue

                                    # 处理不同类型
                                    if isinstance(value, (list, dict)):
                                        processed_value = json.dumps(
                                            value, ensure_ascii=False
                                        )
                                    else:
                                        processed_value = str(value)

                                    update_payload["fields"][col] = processed_value

                            if update_payload["fields"]:
                                records_to_update.append(update_payload)
                                diff_count += 1

                    print(f"    -> 经过差异比较，有 {diff_count} 条记录需要更新")

            print(
                f"    -> 最终准备: {len(records_to_add)} 条新增记录和 {len(records_to_update)} 条更新记录。"
            )

        except FileNotFoundError:
            raise FileNotFoundError(f"找不到调整后的文件 '{edited_filepath}'。")
        except Exception as read_err:
            raise Exception(f"读取调整后文件 '{edited_filepath}' 时出错: {read_err}")

        # --- 2. 执行飞书API操作 --- #
        print(f"\n  步骤 2: 开始执行飞书 API 操作...")

        # 获取access_token
        try:
            access_token = feishu_utils.get_access_token(app_id, app_secret)
            if not access_token:
                raise ValueError("无法获取飞书访问令牌，无法继续同步。")
        except Exception as token_err:
            print(f"     ❌ 获取飞书访问令牌时发生错误: {token_err}")
            results["errors"].append(f"获取飞书令牌失败: {token_err}")
            flash(f"同步失败：无法获取飞书访问令牌。错误: {token_err}", "error")
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"获取飞书访问令牌失败: {token_err}",
                        "summary": results,
                    }
                ),
                500,
            )

        update_table_id = primary_table_ids[0]
        print(f"    目标表格 (更新): {update_table_id}")
        final_add_target_table_id = None
        records_to_add_validated = records_to_add.copy()

        # --- 批量更新 ---
        if records_to_update:
            print(f"      尝试更新 {len(records_to_update)} 条记录...")
            try:
                update_result = feishu_utils.batch_update_records(
                    app_token,
                    update_table_id,
                    records_to_update,
                    app_id,
                    app_secret,
                )
                results["updated"] = update_result.get("success_count", 0)
                results["update_errors"] = update_result.get(
                    "error_count", len(records_to_update) - results["updated"]
                )
                if update_result.get("errors"):
                    results["errors"].extend(
                        [f"更新失败: {e}" for e in update_result["errors"]]
                    )
                print(
                    f"        更新结果: 成功 {results['updated']}, 失败 {results['update_errors']}"
                )
            except Exception as upd_err:
                print(f"     ❌ 调用批量更新时发生错误: {upd_err}")
                results["update_errors"] += len(records_to_update)
                results["errors"].append(f"批量更新API调用失败: {upd_err}")

        # --- 批量新增 (行数检查逻辑) ---
        if records_to_add_validated:
            print(
                f"      开始确定新增目标表格 (共 {len(records_to_add_validated)} 条待新增)..."
            )
            num_to_add = len(records_to_add_validated)
            target_found = False
            if add_target_table_ids:
                print(
                    f"        检测到 {len(add_target_table_ids)} 个指定的新增目标表格，将按顺序检查空间..."
                )
                for target_id in add_target_table_ids:
                    print(f"          > 检查表格 {target_id} 的行数限制...")
                    try:
                        current_count = feishu_utils.get_table_record_count(
                            access_token, app_token, target_id
                        )
                        if current_count is None:
                            print(
                                f"          ⚠️ 无法获取表格 {target_id} 记录数，跳过。"
                            )
                            continue
                        estimated_new_count = current_count + num_to_add
                        print(
                            f"          > 表格 {target_id} 当前: {current_count}, 新增后预估: {estimated_new_count} (上限: {FEISHU_ROW_LIMIT})"
                        )
                        if estimated_new_count <= FEISHU_ROW_LIMIT:
                            print(
                                f"          ✅ 表格 {target_id} 空间足够，选定为新增目标。"
                            )
                            final_add_target_table_id = target_id
                            target_found = True
                            break
                        else:
                            print(f"          ❌ 表格 {target_id} 空间不足。")
                    except Exception as count_err:
                        print(
                            f"          ⚠️ 获取表格 {target_id} 记录数时出错: {count_err}，跳过。"
                        )
                        continue
                if not target_found:
                    error_msg = f"所有指定的新增目标表格 ({', '.join(add_target_table_ids)}) 均已满或无法检查，无法新增 {num_to_add} 条记录。"
                    print(f"     ❌ {error_msg}")
                    results["add_errors"] += num_to_add
                    results["errors"].append(error_msg)
                    records_to_add_validated = []
            else:
                target_id = update_table_id
                print(f"        未指定新增目标表格，将尝试写入主表格: {target_id}")
                print(f"          > 检查表格 {target_id} 的行数限制...")
                try:
                    current_count = feishu_utils.get_table_record_count(
                        access_token, app_token, target_id
                    )
                    if current_count is None:
                        print(
                            f"          ⚠️ 无法获取表格 {target_id} 记录数，将尝试写入，但可能失败。"
                        )
                        final_add_target_table_id = target_id
                        target_found = True
                    else:
                        estimated_new_count = current_count + num_to_add
                        print(
                            f"          > 表格 {target_id} 当前: {current_count}, 新增后预估: {estimated_new_count} (上限: {FEISHU_ROW_LIMIT})"
                        )
                        if estimated_new_count <= FEISHU_ROW_LIMIT:
                            print(
                                f"          ✅ 主表格 {target_id} 空间足够，选定为新增目标。"
                            )
                            final_add_target_table_id = target_id
                            target_found = True
                        else:
                            error_msg = f"主表格 {target_id} 空间不足，无法新增 {num_to_add} 条记录 (当前 {current_count} 行，上限 {FEISHU_ROW_LIMIT})。"
                            print(f"     ❌ {error_msg}")
                            results["add_errors"] += num_to_add
                            results["errors"].append(error_msg)
                            records_to_add_validated = []
                except Exception as count_err:
                    print(
                        f"          ⚠️ 获取主表格 {target_id} 记录数时出错: {count_err}，将尝试写入，但可能失败。"
                    )
                    final_add_target_table_id = target_id
                    target_found = True

            # --- 执行新增 (如果仍有数据且找到目标) ---
            if records_to_add_validated and target_found and final_add_target_table_id:
                print(
                    f"      尝试向表格 {final_add_target_table_id} 新增 {len(records_to_add_validated)} 条记录..."
                )
                try:
                    add_result = feishu_utils.batch_add_records(
                        app_token,
                        final_add_target_table_id,
                        records_to_add_validated,
                        app_id,
                        app_secret,
                    )
                    results["added"] = add_result.get("success_count", 0)
                    results["add_errors"] = add_result.get(
                        "error_count", len(records_to_add_validated) - results["added"]
                    )
                    if add_result.get("errors"):
                        results["errors"].extend(
                            [
                                f"新增失败 (表 {final_add_target_table_id}): {e}"
                                for e in add_result["errors"]
                            ]
                        )
                    print(
                        f"        新增结果: 成功 {results['added']}, 失败 {results['add_errors']}"
                    )
                except Exception as add_err:
                    print(f"     ❌ 调用批量新增时发生错误: {add_err}")
                    results["add_errors"] += len(records_to_add_validated)
                    results["errors"].append(
                        f"批量新增API调用失败 (表 {final_add_target_table_id}): {add_err}"
                    )
            elif records_to_add_validated and not target_found:
                print(f"     ⚠️ 跳过新增：未找到合适的目标表格。")
            elif not records_to_add_validated:
                print(
                    f"     ℹ️ 跳过新增：没有待新增的有效记录（可能因行数限制被阻止）。"
                )

    # --- 统一错误处理 --- #
    except FileNotFoundError as fnf_err:
        print(f"任务 {task_id}: 读取编辑文件时出错: {fnf_err}")
        flash(f"读取文件失败: {fnf_err.filename}", "error")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"读取编辑文件失败: {fnf_err.filename}",
                    "summary": results,
                }
            ),
            500,
        )
    except ValueError as val_err:
        print(f"任务 {task_id}: 数据准备或验证失败: {val_err}")
        flash(f"数据错误: {val_err}", "error")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"数据准备或验证失败: {val_err}",
                    "summary": results,
                }
            ),
            400,
        )
    except Exception as outer_sync_err:
        print(f"任务 {task_id}: 同步过程中发生意外错误: {outer_sync_err}")
        traceback.print_exc()
        results["errors"].append(f"同步过程中发生服务器错误: {outer_sync_err}")
        flash("同步过程中发生内部服务器错误。", "error")
        return (
            jsonify(
                {
                    "success": False,
                    "message": "同步过程中发生内部服务器错误。",
                    "summary": results,
                }
            ),
            500,
        )

    # --- 组装最终响应 --- #
    total_success = results["updated"] + results["added"]
    total_errors = results["update_errors"] + results["add_errors"]
    total_errors += len(
        [
            e
            for e in results["errors"]
            if "调用失败" in e or "表格" in e or "配置" in e or "数据错误" in e
        ]
    )

    success_msg = (
        f"同步完成: 新增 {results['added']} 条, 更新 {results['updated']} 条。"
    )

    # 如果有差异详情，添加到响应中
    if results["diff_details"]:
        results["diff_count"] = len(results["diff_details"])

    if total_errors > 0:
        error_details = []
        if results["update_errors"] > 0:
            error_details.append(f"更新失败 {results['update_errors']} 条")
        if results["add_errors"] > 0:
            error_details.append(f"新增失败 {results['add_errors']} 条")
        other_errors = [e for e in results["errors"] if "调用失败" not in e]
        if other_errors:
            error_details.append(
                f"其他错误 {len(other_errors)} 条: {'; '.join(other_errors[:2])}"
                + ("..." if len(other_errors) > 2 else "")
            )
        error_summary = "部分操作失败: " + "，".join(error_details) + "."
        final_message = success_msg + f" {error_summary}"
        print(f"任务 {task_id}: 同步部分失败。Summary: {results}")
        flash(final_message, "warning")
        return jsonify({"success": True, "message": final_message, "summary": results})
    else:
        print(f"任务 {task_id}: 同步成功。Summary: {results}")
        flash(success_msg, "success")
        return jsonify({"success": True, "message": success_msg, "summary": results})


# Python 标准入口点
if __name__ == "__main__":
    print("启动 Flask 应用...")

    # 加载配置以确保应用启动时变量是最新的
    CURRENT_LLM_CONFIG, CURRENT_FEISHU_CONFIG, CURRENT_POST_PROCESSING_CONFIG = (
        load_config()
    )

    # 再次检查 API Key 并显示警告 (如果需要)
    if not CURRENT_LLM_CONFIG.get("DEEPSEEK_API_KEY") or not CURRENT_LLM_CONFIG.get(
        "DEEPSEEK_API_KEY", ""
    ).startswith("sk-"):
        print("\n********************************************************************")
        print("*** 警告: DEEPSEEK_API_KEY 未设置或格式无效! ***")
        print("***          LLM 处理很可能会失败。             ***")
        print("***          请设置 DEEPSEEK_API_KEY 环境变量   ***")
        print("***          或直接在 app.py 中更新。           ***")
        print("********************************************************************\n")
    # 运行 Flask 开发服务器
    # host='0.0.0.0' 使其在局域网内可访问 (而不仅仅是本机 127.0.0.1)
    # port=5000 指定端口号
    # debug=True 启用调试模式 (代码更改后自动重载，显示详细错误页面)
    # threaded=True 允许多线程处理请求 (对于后台任务是必需的)
    app.run(host="0.0.0.0", port=5100, debug=True, threaded=True)
