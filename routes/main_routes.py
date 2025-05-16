from flask import (
    Blueprint,
    render_template,
    jsonify,
    request,
    send_from_directory,
    current_app,
)
import os
from utils.error_handler import handle_exceptions
from utils.config_manager import load_config
from utils.logger import setup_logger

# 创建Blueprint
main_bp = Blueprint("main", __name__)
logger = setup_logger("main_routes")


@main_bp.route("/")
@handle_exceptions
def index():
    """
    渲染应用程序主页面，提供文件上传和配置设置界面。

    显示Web界面，允许用户上传Excel/CSV文件进行处理，
    配置LLM、飞书和后处理参数，并查看任务处理进度。
    从服务器加载默认配置参数并传递给模板用于初始化界面表单。

    Returns:
        HTML: 渲染后的index.html页面，包含所有配置参数
    """
    logger.info("访问主页 / 请求")
    # 加载当前配置
    llm_config, feishu_config, post_processing_config = load_config()

    return render_template(
        "index.html",
        default_llm_config=llm_config,
        default_feishu_config=feishu_config,
        default_post_processing_cols=post_processing_config,
    )


@main_bp.route("/history")
@handle_exceptions
def history_page():
    """
    显示历史任务记录页面。

    Returns:
        HTML: 渲染后的history.html页面
    """
    logger.info("访问历史页面 /history 请求")
    return render_template("history.html")


@main_bp.route("/static/<path:filename>")
@handle_exceptions
def serve_static(filename):
    """
    提供静态文件服务。

    Args:
        filename: 请求的文件名

    Returns:
        File: 静态文件
    """
    return send_from_directory("static", filename)


def register_routes(app):
    """
    注册所有路由到Flask应用。

    Args:
        app: Flask应用实例
    """
    app.register_blueprint(main_bp)
