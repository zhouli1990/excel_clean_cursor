from flask import Blueprint, request, jsonify
import json
from utils.error_handler import handle_exceptions
from utils.config_manager import load_config, save_config, validate_config

# 创建Blueprint
config_bp = Blueprint("config", __name__)


@config_bp.route("/save_config", methods=["POST"])
@handle_exceptions
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

    # 验证配置数据
    is_valid, error_msg = validate_config(new_config_data)
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    # 加载当前配置以获取默认值
    current_llm_config, current_feishu_config, current_post_processing_config = (
        load_config()
    )

    # 确保"来源"列在目标列中
    if "TARGET_COLUMNS" in new_config_data.get("llm_config", {}):
        target_columns = new_config_data["llm_config"]["TARGET_COLUMNS"]
        if "来源" not in target_columns:
            target_columns.append("来源")

    # 保存配置
    if save_config(new_config_data):
        return jsonify({"success": True, "message": "配置已成功保存为默认值！"})
    else:
        return (
            jsonify({"success": False, "error": "保存配置文件时发生服务器错误。"}),
            500,
        )


def register_routes(app):
    """
    注册所有路由到Flask应用。

    Args:
        app: Flask应用实例
    """
    app.register_blueprint(config_bp)
