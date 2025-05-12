from flask import Blueprint, request, jsonify
import os
import json
import time
from utils.error_handler import handle_exceptions
from utils.task_manager import get_task_info, update_task_history_entry
from utils.config_manager import load_config
import pandas as pd
import feishu_utils
from datetime import datetime

# 创建Blueprint
feishu_bp = Blueprint("feishu", __name__)


@feishu_bp.route("/check_diff/<task_id>", methods=["GET"])
@handle_exceptions
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
    task_info = get_task_info(task_id)
    if not task_info:
        return jsonify({"success": False, "error": "任务ID不存在或已过期。"}), 404

    # 检查必要的文件是否存在
    final_filename = task_info.get("result_file")  # 最终文件名

    if not final_filename:
        return jsonify({"success": False, "error": "找不到处理结果文件信息。"}), 404

    from flask import current_app

    output_dir = os.path.join(current_app.config["OUTPUT_FOLDER"], task_id)
    final_filepath = os.path.join(output_dir, final_filename)

    if not os.path.exists(final_filepath):
        return jsonify({"success": False, "error": "找不到多Sheet页Excel文件。"}), 404

    try:
        # 读取Excel文件的各个Sheet
        sheet_info = {}
        total_rows = 0
        all_columns = set()
        record_id_col = "record_id"

        # 加载当前配置
        _, feishu_config, _ = load_config()
        phone_col = feishu_config.get("PHONE_NUMBER_COLUMN", "电话")

        with pd.ExcelFile(final_filepath) as xls:
            sheets = xls.sheet_names

            for sheet_name in sheets:
                # 读取每个Sheet
                converters = {phone_col: str}
                df = pd.read_excel(xls, sheet_name=sheet_name, converters=converters)
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

        return jsonify(result)

    except Exception as e:
        return (
            jsonify(
                {"success": False, "error": f"读取Excel文件时发生服务器错误: {str(e)}"}
            ),
            500,
        )


@feishu_bp.route("/sync_to_feishu/<task_id>", methods=["POST"])
@handle_exceptions
def sync_to_feishu(task_id):
    """
    将数据同步到飞书多维表格。

    基于多Sheet页Excel文件或直接导入的源文件执行同步操作:
    1. 多Sheet页Excel - 从"新增"和"更新"Sheet页读取数据
    2. 直接导入文件 - 读取单个Sheet页，视为全部新增

    执行前会验证配置、检查飞书表空间容量并格式化数据，
    同步结果会详细记录成功和失败的操作数量。

    Path Params:
        - task_id: 任务唯一标识符UUID

    Returns:
        JSON: 同步操作结果，包含添加/更新记录数和错误信息
    """
    # --- 常量定义 ---
    FEISHU_ROW_LIMIT = 50000

    task_info = get_task_info(task_id)
    if not task_info:
        return jsonify({"success": False, "error": "任务ID不存在或已过期"}), 404

    # --- 获取配置 ---
    # 加载当前配置
    _, feishu_config, _ = load_config()

    # 如果任务中有配置信息，使用任务配置
    task_feishu_config = task_info.get("config", {}).get("feishu_config", {})
    if task_feishu_config:
        feishu_config.update(task_feishu_config)

    app_id = feishu_config.get("APP_ID")
    app_secret = feishu_config.get("APP_SECRET")
    app_token = feishu_config.get("APP_TOKEN")
    primary_table_ids = feishu_config.get("TABLE_IDS", [])
    add_target_table_ids = feishu_config.get("ADD_TARGET_TABLE_IDS", [])

    # 如果没有添加目标表，使用主表
    if not add_target_table_ids and primary_table_ids:
        add_target_table_ids = primary_table_ids.copy()

    # --- 检查必需的信息和配置 --- #
    # 判断同步模式
    is_direct_import = task_info.get("direct_import", False)

    # 1. 优先使用用户上传的原始文件
    if "direct_import_source_file_path" in task_info and os.path.exists(
        task_info["direct_import_source_file_path"]
    ):
        file_to_sync = task_info["direct_import_source_file_path"]
        file_source_type = "用户上传原始文件"
    elif "edited_file_path" in task_info and os.path.exists(
        task_info["edited_file_path"]
    ):
        file_to_sync = task_info["edited_file_path"]
        file_source_type = "处理结果文件"
    else:
        return jsonify({"success": False, "error": "找不到任何可同步的文件"}), 400

    print(
        f"[DEBUG] 实际用于同步的文件路径: {file_to_sync}，来源类型: {file_source_type}"
    )

    # 2. 验证API配置
    if not app_id or not app_secret or not app_token:
        return (
            jsonify({"success": False, "error": "配置文件缺少必要的飞书API凭据"}),
            400,
        )

    if not primary_table_ids and not add_target_table_ids:
        return jsonify({"success": False, "error": "配置文件未指定任何表格ID"}), 400

    if not os.path.exists(file_to_sync):
        return (
            jsonify({"success": False, "error": f"同步文件不存在: {file_to_sync}"}),
            400,
        )

    # --- 开始执行同步流程 --- #
    records_to_update = []
    records_to_add = []
    results = {
        "updated": 0,
        "added": 0,
        "update_errors": 0,
        "add_errors": 0,
        "errors": [],
        "details": [],
    }

    try:
        # --- 1. 读取文件数据 --- #
        target_columns = (
            task_info.get("config", {}).get("llm_config", {}).get("TARGET_COLUMNS", [])
        )

        print(f"[DEBUG] 当前 target_columns: {target_columns}")
        print(f"[DEBUG] 当前任务ID: {task_id}")
        print(f"[DEBUG] 同步文件路径: {file_to_sync}")
        print(f"[DEBUG] 是否直接导入模式: {is_direct_import}")
        # 根据不同的同步模式处理数据
        if is_direct_import:
            # 只同步"新增"Sheet
            try:
                try:
                    df = pd.read_excel(file_to_sync, sheet_name="新增")
                    print(
                        f"[DEBUG] 读取Sheet: 新增，行数: {len(df)}，列: {list(df.columns)}"
                    )
                except ValueError as e:
                    print(f"[ERROR] 未找到Sheet: 新增。错误: {e}")
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "未找到Sheet: 新增，请检查上传文件。",
                            }
                        ),
                        400,
                    )
                if df.empty:
                    print("[WARN] 新增Sheet为空，无数据可同步。")
                    return (
                        jsonify(
                            {"success": False, "error": "新增Sheet为空，无数据可同步。"}
                        ),
                        400,
                    )
                # 准备添加记录
                feishu_id_col = "record_id"
                if feishu_id_col not in df.columns:
                    df[feishu_id_col] = ""
                df = df.fillna("")
                records_to_add = []
                for _, row in df.iterrows():
                    add_payload = {"fields": {}}
                    for col in df.columns:
                        if (
                            col == feishu_id_col
                            or col == "local_row_id"
                            or col == "行ID"
                            or col == "row_id"
                        ):
                            continue
                        if target_columns and col not in target_columns:
                            continue
                        value = row[col]
                        if pd.isna(value) or (
                            isinstance(value, str) and not value.strip()
                        ):
                            continue
                        if isinstance(value, (list, dict)):
                            value_str = json.dumps(value, ensure_ascii=False)
                        else:
                            value_str = str(value)
                        add_payload["fields"][col] = value_str
                    if add_payload["fields"]:
                        records_to_add.append(add_payload)
                print(
                    f"[DEBUG] 直接导入模式，最终准备同步的新增记录数: {len(records_to_add)}"
                )

            except Exception as read_err:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"读取直接导入源文件时出错: {str(read_err)}",
                        }
                    ),
                    500,
                )

        else:
            # 多Sheet页Excel模式 - 分别处理"新增"和"更新"Sheet
            try:
                with pd.ExcelFile(file_to_sync) as xls:
                    sheets = xls.sheet_names
                    print(f"[DEBUG] Excel文件包含Sheet: {sheets}")
                    # 处理"新增"Sheet
                    if "新增" in sheets:
                        df_add = pd.read_excel(xls, sheet_name="新增")
                        print(f"[DEBUG] 读取'新增'Sheet数据行数: {len(df_add)}")
                        feishu_id_col = "record_id"
                        if feishu_id_col not in df_add.columns:
                            df_add[feishu_id_col] = ""
                        else:
                            df_add[feishu_id_col] = (
                                df_add[feishu_id_col].fillna("").astype(str).str.strip()
                            )
                        df_add = df_add[df_add[feishu_id_col] == ""]
                        records_to_add = []
                        for _, row in df_add.iterrows():
                            add_payload = {"fields": {}}
                            for col in df_add.columns:
                                if (
                                    col == feishu_id_col
                                    or col == "local_row_id"
                                    or col == "行ID"
                                    or col == "row_id"
                                ):
                                    continue
                                if target_columns and col not in target_columns:
                                    continue
                                value = row[col]
                                if pd.isna(value) or (
                                    isinstance(value, str) and not value.strip()
                                ):
                                    continue
                                if isinstance(value, (list, dict)):
                                    value_str = json.dumps(value, ensure_ascii=False)
                                else:
                                    value_str = str(value)
                                add_payload["fields"][col] = value_str
                            if add_payload["fields"]:
                                records_to_add.append(add_payload)
                        print(
                            f"[DEBUG] Sheet模式，最终准备同步的新增记录数: {len(records_to_add)}"
                        )

                    # 处理"更新"Sheet
                    if "更新" in sheets:
                        df_update = pd.read_excel(xls, sheet_name="更新")

                        # 确保record_id列存在且非空
                        feishu_id_col = "record_id"
                        if feishu_id_col not in df_update.columns:
                            # 没有record_id列，跳过更新处理
                            pass
                        else:
                            # 清理record_id
                            df_update[feishu_id_col] = (
                                df_update[feishu_id_col]
                                .fillna("")
                                .astype(str)
                                .str.strip()
                            )

                            # 筛选有效record_id的行
                            df_update_valid = df_update[df_update[feishu_id_col] != ""]

                            # 准备更新数据
                            for _, row in df_update_valid.iterrows():
                                record_id = row[feishu_id_col]
                                update_payload = {"record_id": record_id, "fields": {}}
                                for col in df_update_valid.columns:
                                    if (
                                        col == feishu_id_col
                                        or col == "local_row_id"
                                        or col == "行ID"
                                        or col == "row_id"
                                    ):
                                        continue
                                    if target_columns and col not in target_columns:
                                        continue
                                    value = row[col]
                                    if pd.isna(value) or (
                                        isinstance(value, str) and not value.strip()
                                    ):
                                        continue
                                    if isinstance(value, (list, dict)):
                                        value_str = json.dumps(
                                            value, ensure_ascii=False
                                        )
                                    else:
                                        value_str = str(value)
                                    update_payload["fields"][col] = value_str
                                if update_payload["fields"]:
                                    print(
                                        f"[DEBUG] 更新payload字段: {list(update_payload['fields'].keys())}"
                                    )
                                    records_to_update.append(update_payload)

            except Exception as read_err:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"读取多Sheet页Excel文件时出错: {str(read_err)}",
                        }
                    ),
                    500,
                )

        # --- 2. 执行飞书同步操作 --- #

        # 获取访问令牌
        try:
            access_token = feishu_utils.get_access_token(app_id, app_secret)
            if not access_token:
                return jsonify({"success": False, "error": "获取飞书访问令牌失败"}), 500
        except Exception as token_err:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"获取飞书访问令牌时出错: {str(token_err)}",
                    }
                ),
                500,
            )

        # 检查所有目标表格的容量，确认可以容纳新增数据
        if records_to_add:
            records_count = len(records_to_add)
            target_table_id = None

            for table_id in add_target_table_ids:
                try:
                    # 获取表格当前记录数
                    current_count = feishu_utils.get_table_record_count(
                        access_token, app_token, table_id
                    )
                    available_space = FEISHU_ROW_LIMIT - current_count

                    if available_space >= records_count:
                        target_table_id = table_id
                        break

                except Exception:
                    continue

            if not target_table_id:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"所有目标表格都已达到或接近行数限制({FEISHU_ROW_LIMIT})，无法添加{records_count}条新记录",
                        }
                    ),
                    400,
                )

            # 执行批量新增
            try:
                add_results = feishu_utils.batch_add_records(
                    app_token, target_table_id, records_to_add, app_id, app_secret
                )

                results["added"] = add_results.get("success_count", 0)
                results["add_errors"] = add_results.get("error_count", 0)
                if "errors" in add_results:
                    results["errors"].extend(add_results["errors"])

            except Exception as add_err:
                results["errors"].append(f"批量新增记录时出错: {str(add_err)}")

        # 执行批量更新
        if records_to_update and primary_table_ids:
            # 选择第一个主表ID作为更新目标
            update_table_id = primary_table_ids[0]

            try:
                update_results = feishu_utils.batch_update_records(
                    app_token, update_table_id, records_to_update, app_id, app_secret
                )

                results["updated"] = update_results.get("success_count", 0)
                results["update_errors"] = update_results.get("error_count", 0)
                if "errors" in update_results:
                    results["errors"].extend(update_results["errors"])

            except Exception as update_err:
                results["errors"].append(f"批量更新记录时出错: {str(update_err)}")

        # 更新任务状态
        task_info["sync_result"] = results
        task_info["status"] = "同步完成"

        # 构建摘要信息
        summary = f"新增: {results['added']}条, 更新: {results['updated']}条"
        if results["add_errors"] > 0 or results["update_errors"] > 0:
            summary += f", 失败: {results['add_errors'] + results['update_errors']}条"

        # 更新历史记录
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_task_history_entry(
            task_id, {"status": "成功", "completion_time": now, "sync_status": summary}
        )

        # 返回结果
        return jsonify(
            {"success": True, "message": f"同步完成! {summary}", "results": results}
        )

    except Exception as e:
        error_msg = f"同步过程中发生错误: {str(e)}"

        # 更新历史记录
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_task_history_entry(
            task_id,
            {"status": "失败", "completion_time": now, "error_message": error_msg},
        )

        return jsonify({"success": False, "error": error_msg}), 500


def register_routes(app):
    """
    注册所有路由到Flask应用。

    Args:
        app: Flask应用实例
    """
    app.register_blueprint(feishu_bp)
