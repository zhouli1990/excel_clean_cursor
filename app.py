import os
import logging
from flask import Flask
from utils.logger import setup_logger
from utils.config_manager import load_config
from utils.task_manager import clean_up_old_tasks
from routes.main_routes import register_routes as register_main_routes
from routes.task_routes import register_routes as register_task_routes
from routes.file_routes import register_routes as register_file_routes
from routes.config_routes import register_routes as register_config_routes
from routes.feishu_routes import register_routes as register_feishu_routes

# 设置应用级别日志记录器
logger = setup_logger("app", level="info", detailed_format=True)


def create_app():
    """
    创建并配置Flask应用实例。

    Returns:
        Flask: 配置好的Flask应用实例
    """
    # 初始化 Flask 应用
    app = Flask(__name__)

    # --- 设置 Secret Key --- #
    # 对于生产环境，强烈建议从环境变量或配置文件读取一个固定且保密的密钥
    # 例如: app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_dev_key')
    # 为了方便本地运行，这里使用 os.urandom 生成一个随机密钥
    # 注意：每次重启服务密钥都会改变，这会导致之前的 session 失效
    app.secret_key = os.urandom(24)
    logger.info("Flask app secret key 已随机设置")

    # 配置上传文件存储路径和输出路径
    app.config["UPLOAD_FOLDER"] = "uploads"
    app.config["OUTPUT_FOLDER"] = "outputs"

    # 确保上传和输出目录存在
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    # 加载配置
    load_config()

    # 注册所有路由
    register_routes(app)

    # 清理旧任务
    clean_up_old_tasks(days_to_keep=7)

    return app


def register_routes(app):
    """
    注册所有蓝图路由到Flask应用。

    Args:
        app: Flask应用实例
    """
    # 注册各模块的路由
    register_main_routes(app)
    register_task_routes(app)
    register_file_routes(app)
    register_config_routes(app)
    register_feishu_routes(app)

    logger.info("所有路由已注册")


# 主入口点
if __name__ == "__main__":
    logger.info("启动Flask应用...")
    app = create_app()

    # 检查API Key并显示警告
    llm_config, _, _ = load_config()
    if not llm_config.get("DEEPSEEK_API_KEY") or not llm_config.get(
        "DEEPSEEK_API_KEY", ""
    ).startswith("sk-"):
        logger.warning("警告: DEEPSEEK_API_KEY 未设置或格式无效! LLM处理很可能会失败。")

    # 运行Flask开发服务器
    app.run(host="0.0.0.0", port=5100, debug=True, threaded=True)
