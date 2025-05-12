from flask import Blueprint, request, jsonify, send_from_directory, current_app
import os
import json
import time
import threading
import pandas as pd
from werkzeug.utils import secure_filename
from utils.error_handler import handle_exceptions
from utils.task_manager import create_task, get_task_info, get_latest_task
from utils.config_manager import load_config
from routes.task_routes import run_processing
from postprocessor import apply_post_processing, create_multi_sheet_excel

# 创建Blueprint
file_bp = Blueprint("file", __name__)


@file_bp.route("/upload", methods=["POST"])
@handle_exceptions
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

    # 为本次任务生成唯一ID并创建任务
    task_id = create_task(
        files=[f.filename for f in files], task_type="complete_processing"
    )

    # 根据task_id创建独立的上传和输出子目录
    upload_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], task_id)
    output_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], task_id)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_file_paths = []
    try:
        # 遍历上传的文件列表
        for file in files:
            # 确保文件存在且有文件名
            if file and file.filename:
                # 直接使用原始文件名，不再使用secure_filename处理
                filename = file.filename
                print(f"使用原始文件名保存: {filename}")  # 添加日志
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
            return jsonify({"error": "没有有效的上传文件"}), 400

        # --- 从前端请求中获取配置参数 ---
        task_config = parse_config_from_request(request)

        # 更新任务信息中的配置
        task_info = get_task_info(task_id)
        if task_info:
            task_info["config"] = task_config
            task_info["input_files"] = input_file_paths

        # 定义输出的Excel文件名
        output_filename = f"consolidated_{task_id}.xlsx"
        # 构造完整的输出文件路径
        output_filepath = os.path.join(output_dir, output_filename)

        # 创建并启动一个后台线程来执行实际的数据处理
        thread = threading.Thread(
            target=run_processing,
            args=(task_id, input_file_paths, output_filepath, task_config),
        )
        thread.start()

        # 返回任务ID给前端，以便后续查询进度
        return jsonify({"task_id": task_id})

    except Exception as e:
        # 返回错误信息
        return jsonify({"error": f"保存上传文件失败: {str(e)}"}), 500


@file_bp.route("/download/<task_id>/<filename>")
@handle_exceptions
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
    directory = os.path.join(current_app.config["OUTPUT_FOLDER"], task_id)

    # 检查输出目录是否存在
    if not os.path.exists(directory):
        return jsonify({"error": "任务输出目录未找到"}), 404

    # 安全性检查: 防止路径遍历攻击
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "文件名无效"}), 400

    # a构造完整的文件路径
    file_path = os.path.join(directory, filename)

    # 检查文件是否存在
    if not os.path.isfile(file_path):
        return jsonify({"error": "结果文件未找到"}), 404

    # 使用Flask的send_from_directory函数安全地发送文件
    # as_attachment=True 会让浏览器弹出下载对话框
    return send_from_directory(directory, filename, as_attachment=True)


@file_bp.route("/upload_edited/<task_id>", methods=["POST"])
@handle_exceptions
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
    task_info = get_task_info(task_id)
    if not task_info:
        return jsonify({"success": False, "error": "任务ID不存在或已过期。"}), 404

    # 检查是否有文件上传
    if "edited_file" not in request.files:
        return (
            jsonify({"success": False, "error": "请求中缺少'edited_file'文件部分。"}),
            400,
        )

    file = request.files["edited_file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "未选择任何文件。"}), 400

    if file:  # 再次确认文件存在
        # 构建保存路径(保存在该任务的输出目录下)
        output_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], task_id)
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始文件名作为参考
        original_filename = file.filename
        print(f"上传的编辑后文件原始文件名: {original_filename}")

        edited_filename = f"edited_{task_id}.xlsx"
        edited_filepath = os.path.join(output_dir, edited_filename)

        try:
            # 保存文件
            file.save(edited_filepath)
            # 更新任务状态，表示已收到修改后的文件
            task_info["edited_file_uploaded"] = True
            task_info["edited_file_path"] = edited_filepath
            task_info["original_edited_filename"] = original_filename  # 保存原始文件名

            return jsonify(
                {
                    "success": True,
                    "message": "调整后的文件上传成功！",
                    "edited_filename": edited_filename,
                }
            )
        except Exception as e:
            return (
                jsonify(
                    {"success": False, "error": f"保存文件时发生服务器错误: {str(e)}"}
                ),
                500,
            )

    return jsonify({"success": False, "error": "上传失败，未知错误。"}), 500


@file_bp.route("/upload_new_or_associate_file", methods=["POST"])
@handle_exceptions
def upload_new_or_associate_file():
    """
    通用的文件上传处理路由，既能处理已编辑文件的上传，也能处理全新文件的直接导入。

    如果存在已完成预处理的任务ID，则将上传文件作为该任务的编辑文件处理；
    否则，创建新任务作为直接导入飞书的源文件。

    Request:
        - file: 用户上传的Excel文件

    Returns:
        JSON: 上传成功或失败信息，以及关联的任务ID
    """
    # 检查是否有文件上传
    if "file" not in request.files:
        return jsonify({"success": False, "error": "请求中缺少文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "未选择任何文件"}), 400

    # 检查文件类型
    if not (file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
        return (
            jsonify({"success": False, "error": "只支持上传Excel文件(.xlsx或.xls)"}),
            400,
        )

    # 获取最近完成的预处理任务ID
    latest_task_id = get_latest_task()

    # 情况1: 有有效的最近预处理任务，将文件作为该任务的编辑文件处理
    if latest_task_id and get_task_info(latest_task_id):
        task_id = latest_task_id

        # 创建输出目录（如果不存在）
        output_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], task_id)
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始文件名作为参考
        original_filename = file.filename
        print(f"关联到已有任务的文件原始文件名: {original_filename}")

        # 生成编辑后文件名和保存路径
        edited_filename = f"edited_{task_id}.xlsx"
        edited_file_path = os.path.join(output_dir, edited_filename)

        try:
            # 保存上传的文件
            file.save(edited_file_path)

            # 更新任务信息
            task_info = get_task_info(task_id)
            if task_info:
                task_info["edited_file_path"] = edited_file_path
                task_info["edited_file_uploaded"] = True
                task_info["original_edited_filename"] = (
                    original_filename  # 保存原始文件名
                )

            return jsonify(
                {
                    "success": True,
                    "message": f"文件已成功上传并关联到任务{task_id}",
                    "task_id": task_id,
                    "is_new_task": False,
                }
            )

        except Exception as e:
            return (
                jsonify({"success": False, "error": f"保存编辑后文件时出错: {str(e)}"}),
                500,
            )

    # 情况2: 没有有效的最近预处理任务，创建新任务ID并处理为直接导入源文件
    else:
        # 为新任务创建ID
        task_id = create_task(files=[file.filename], task_type="direct_import")

        # 创建任务目录
        upload_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], task_id)
        output_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], task_id)
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始文件名作为参考
        original_filename = file.filename
        print(f"创建新任务的文件原始文件名: {original_filename}")

        # 保存为直接导入源文件，使用原始文件名
        direct_import_file_path = os.path.join(upload_dir, original_filename)

        # 解析目标列配置（TARGET_COLUMNS）
        target_columns_json = request.form.get("config_target_columns")
        target_columns = []
        if target_columns_json:
            try:
                target_columns = json.loads(target_columns_json)
                print(f"[DEBUG] 前端传递的TARGET_COLUMNS: {target_columns}")
            except Exception as e:
                print(f"[WARN] 解析前端TARGET_COLUMNS失败: {e}")
                target_columns = []
        # 兜底：如未传递则从默认配置加载
        if not target_columns:
            from utils.config_manager import load_config

            llm_config, _, _ = load_config()
            target_columns = llm_config.get("TARGET_COLUMNS", [])
            print(f"[DEBUG] 兜底加载配置文件中的TARGET_COLUMNS: {target_columns}")

        try:
            # 保存上传的文件
            file.save(direct_import_file_path)

            # 更新任务信息
            task_info = get_task_info(task_id)
            if task_info:
                task_info["direct_import_source_file_path"] = direct_import_file_path
                task_info["direct_import"] = True
                task_info["original_filename"] = original_filename  # 保存原始文件名
                # 写入目标列配置
                if "config" not in task_info:
                    task_info["config"] = {}
                if "llm_config" not in task_info["config"]:
                    task_info["config"]["llm_config"] = {}
                task_info["config"]["llm_config"]["TARGET_COLUMNS"] = target_columns
                print(f"[DEBUG] 已写入任务的TARGET_COLUMNS: {target_columns}")

            return jsonify(
                {
                    "success": True,
                    "message": f"文件已成功上传并创建新任务{task_id}用于直接导入",
                    "task_id": task_id,
                    "is_new_task": True,
                }
            )

        except Exception as e:
            return (
                jsonify(
                    {"success": False, "error": f"保存直接导入文件时出错: {str(e)}"}
                ),
                500,
            )


def parse_config_from_request(request):
    """
    从请求中解析配置参数。

    Args:
        request: Flask请求对象

    Returns:
        dict: 配置字典
    """
    # 加载当前默认配置作为基础
    llm_config, feishu_config, post_processing_config = load_config()

    try:
        # LLM配置
        target_columns_json = request.form.get("config_target_columns", "[]")
        target_columns = json.loads(target_columns_json)

        # 验证目标列列表格式
        if not isinstance(target_columns, list) or not all(
            isinstance(s, str) for s in target_columns
        ):
            target_columns = llm_config["TARGET_COLUMNS"]

        # 确保"来源"列包含在任务配置中
        source_column_name = "来源"
        if source_column_name not in target_columns:
            target_columns.append(source_column_name)

        # API密钥
        api_key = request.form.get(
            "config_api_key", llm_config["DEEPSEEK_API_KEY"]
        ).strip()
        if not api_key:
            api_key = llm_config["DEEPSEEK_API_KEY"]

        # 数值类型参数
        try:
            batch_size = int(request.form.get("config_batch_size"))
        except (ValueError, TypeError):
            batch_size = llm_config["BATCH_SIZE"]

        try:
            max_tokens = int(request.form.get("config_max_tokens"))
        except (ValueError, TypeError):
            max_tokens = llm_config["MAX_COMPLETION_TOKENS"]

        try:
            api_timeout = int(request.form.get("config_api_timeout"))
        except (ValueError, TypeError):
            api_timeout = llm_config["API_TIMEOUT"]

        # 百炼API相关配置
        dashscope_api_key = request.form.get(
            "config_dashscope_api_key", llm_config.get("DASHSCOPE_API_KEY", "")
        ).strip()

        bailian_model_name = request.form.get(
            "config_bailian_model_name",
            llm_config.get("BAILIAN_MODEL_NAME", "qwen-turbo-latest"),
        ).strip()

        bailian_completion_window = request.form.get(
            "config_bailian_completion_window",
            llm_config.get("BAILIAN_COMPLETION_WINDOW", "24h"),
        ).strip()

        # 创建动态配置字典
        dynamic_config = {
            "TARGET_COLUMNS": target_columns,  # 目标列，默认值见DEFAULT_PROCESSOR_CONFIG
            "DEEPSEEK_API_KEY": api_key,  # DeepSeek API Key
            "DEEPSEEK_API_ENDPOINT": llm_config.get(
                "DEEPSEEK_API_ENDPOINT",
                "https://api.deepseek.com/chat/completions",  # 默认DeepSeek API端点
            ),
            "BATCH_SIZE": batch_size,  # 批处理大小，默认50
            "MAX_COMPLETION_TOKENS": max_tokens,  # 最大token数，默认8192
            "API_TIMEOUT": api_timeout,  # 超时，默认180
            "DASHSCOPE_API_KEY": dashscope_api_key,  # 百炼API Key
            "BAILIAN_MODEL_NAME": bailian_model_name,  # 百炼模型名
            "BAILIAN_COMPLETION_WINDOW": bailian_completion_window,  # 百炼窗口
            "BAILIAN_BASE_URL": llm_config.get(
                "BAILIAN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 默认百炼API地址
            ),
            "BAILIAN_BATCH_ENDPOINT": llm_config.get(
                "BAILIAN_BATCH_ENDPOINT", "/v1/chat/completions"  # 默认百炼批处理端点
            ),
        }

        # 飞书配置
        feishu_app_id = request.form.get(
            "feishu_app_id", feishu_config.get("APP_ID", "")  # 飞书AppId，默认空
        ).strip()
        feishu_app_secret = request.form.get(
            "feishu_app_secret",
            feishu_config.get("APP_SECRET", ""),  # 飞书AppSecret，默认空
        ).strip()
        feishu_app_token = request.form.get(
            "feishu_app_token",
            feishu_config.get("APP_TOKEN", ""),  # 飞书AppToken，默认空
        ).strip()
        feishu_table_ids_json = request.form.get("feishu_table_ids", "[]")

        try:
            feishu_table_ids = json.loads(feishu_table_ids_json)
            if not isinstance(feishu_table_ids, list):
                feishu_table_ids = feishu_config["TABLE_IDS"]
        except json.JSONDecodeError:
            feishu_table_ids = feishu_config["TABLE_IDS"]

        # 后处理配置
        post_process_config_json = request.form.get("post_process_config", "{}")
        try:
            post_process_config = json.loads(post_process_config_json)
            if not isinstance(post_process_config, dict):
                post_process_config = {}
        except json.JSONDecodeError:
            post_process_config = {}

        # 合并所有配置到一个字典
        task_config = {
            "llm_config": dynamic_config,
            "feishu_config": {
                "APP_ID": feishu_app_id,
                "APP_SECRET": feishu_app_secret,
                "APP_TOKEN": feishu_app_token,
                "TABLE_IDS": feishu_table_ids,
                "ADD_TARGET_TABLE_IDS": feishu_config.get(
                    "ADD_TARGET_TABLE_IDS", []
                ),  # 默认空列表
                "COMPANY_NAME_COLUMN": feishu_config.get(
                    "COMPANY_NAME_COLUMN", "企业名称"
                ),  # 默认"企业名称"
                "PHONE_NUMBER_COLUMN": feishu_config.get(
                    "PHONE_NUMBER_COLUMN", "电话"
                ),  # 默认"电话"
                "REMARK_COLUMN_NAME": feishu_config.get(
                    "REMARK_COLUMN_NAME", "备注"
                ),  # 默认"备注"
                "RELATED_COMPANY_COLUMN_NAME": feishu_config.get(
                    "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
                ),  # 默认"关联公司名称(LLM)"
            },
            "post_processing_config": post_process_config,
        }

        return task_config

    except (ValueError, TypeError, json.JSONDecodeError) as config_err:
        # 如果解析出错，回退到默认配置
        return {
            "llm_config": llm_config,
            "feishu_config": feishu_config,
            "post_processing_config": post_processing_config,
        }


def register_routes(app):
    """
    注册所有路由到Flask应用。

    Args:
        app: Flask应用实例
    """
    app.register_blueprint(file_bp)
