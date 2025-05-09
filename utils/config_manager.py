import os
import json
import logging
from typing import Dict, Tuple, Any

# 从logger.py导入统一的日志工具
from utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger("config_manager")

# 定义配置文件路径
CONFIG_FILE = "config.json"

# --- 默认配置 ---
# 数据处理模块的默认配置
DEFAULT_PROCESSOR_CONFIG = {
    # 定义最终输出 Excel 文件需要包含的目标列名
    "TARGET_COLUMNS": [
        "公司名称",
        "联系人",
        "职位",
        "电话",
        "来源",
    ],
    # 从环境变量获取 DeepSeek API Key，如果未设置则使用默认值
    "DEEPSEEK_API_KEY": os.environ.get(
        "DEEPSEEK_API_KEY",
        "sk-9df835c781904b2289663567d05c94d9",
    ),
    # DeepSeek API 的接入点
    "DEEPSEEK_API_ENDPOINT": "https://api.deepseek.com/chat/completions",
    # 调用 LLM API 时，每次处理的行数 (批处理大小)
    "BATCH_SIZE": 160,
    # LLM API 调用时，允许生成的最大 token 数量
    "MAX_COMPLETION_TOKENS": 8192,
    "API_TIMEOUT": 180,  # 默认API超时
}

# 飞书配置默认值
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
    "ADD_TARGET_TABLE_IDS": [],
    "COMPANY_NAME_COLUMN": "企业名称",
    "PHONE_NUMBER_COLUMN": "电话",
    "REMARK_COLUMN_NAME": "备注",
    "RELATED_COMPANY_COLUMN_NAME": "关联公司名称(LLM)",
}

# 后处理默认配置
DEFAULT_POST_PROCESSING_COLS = {
    "check_duplicate_phones": {"post_phone_col_for_dup_phones": "电话"},
    "check_duplicate_companies": {
        "post_phone_col_for_dup_comp": "电话",
        "post_company_col_for_dup_comp": "公司名称",
    },
}


def load_config() -> Tuple[Dict, Dict, Dict]:
    """
    尝试从 config.json 加载配置，如果失败则使用默认值并保存。

    Returns:
        Tuple[Dict, Dict, Dict]: LLM配置, 飞书配置, 后处理配置
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_from_file = json.load(f)

                # 验证配置格式
                if (
                    "llm_config" in config_from_file
                    and "feishu_config" in config_from_file
                    and "post_processing_config" in config_from_file
                ):
                    logger.info(f"成功从 {CONFIG_FILE} 加载配置。")

                    # 确保飞书配置包含ADD_TARGET_TABLE_IDS
                    loaded_feishu_config = config_from_file["feishu_config"]
                    if "ADD_TARGET_TABLE_IDS" not in loaded_feishu_config:
                        loaded_feishu_config["ADD_TARGET_TABLE_IDS"] = (
                            DEFAULT_FEISHU_CONFIG["ADD_TARGET_TABLE_IDS"]
                        )

                    # 确保目标列包含"来源"
                    llm_config = config_from_file["llm_config"]
                    if "来源" not in llm_config.get("TARGET_COLUMNS", []):
                        logger.info(
                            "配置加载后：将缺失的 '来源' 添加到 TARGET_COLUMNS。"
                        )
                        llm_config.setdefault("TARGET_COLUMNS", []).append("来源")

                    # 返回加载的配置
                    return (
                        llm_config,
                        loaded_feishu_config,
                        config_from_file.get(
                            "post_processing_config",
                            DEFAULT_POST_PROCESSING_COLS.copy(),
                        ),
                    )
                else:
                    logger.warning(
                        f"警告: {CONFIG_FILE} 文件格式不完整，使用默认配置。"
                    )
        else:
            logger.info(f"配置文件 {CONFIG_FILE} 不存在，使用默认配置。")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"加载配置文件 {CONFIG_FILE} 时出错: {e}，使用默认配置。")

    # 如果加载失败或文件不存在，使用默认值并尝试保存
    llm_config = DEFAULT_PROCESSOR_CONFIG.copy()
    feishu_config = DEFAULT_FEISHU_CONFIG.copy()
    post_processing_config = DEFAULT_POST_PROCESSING_COLS.copy()

    # 确保"来源"在目标列中
    if "来源" not in llm_config["TARGET_COLUMNS"]:
        llm_config["TARGET_COLUMNS"].append("来源")

    # 尝试写入默认配置
    default_config = {
        "llm_config": llm_config,
        "feishu_config": feishu_config,
        "post_processing_config": post_processing_config,
    }
    save_config(default_config)

    return llm_config, feishu_config, post_processing_config


def save_config(config_data: Dict[str, Any]) -> bool:
    """
    将配置数据写入 config.json 文件。

    Args:
        config_data: 包含所有配置的字典

    Returns:
        bool: 是否成功保存
    """
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        logger.info(f"配置已成功保存到 {CONFIG_FILE}。")
        return True
    except IOError as e:
        logger.error(f"保存配置文件 {CONFIG_FILE} 时出错: {e}")
        return False


def validate_config(config: Dict) -> Tuple[bool, str]:
    """
    验证配置项的正确性。

    Args:
        config: 要验证的配置字典

    Returns:
        Tuple[bool, str]: (验证是否通过, 错误消息)
    """
    if not isinstance(config, dict):
        return False, "配置必须是字典类型"

    # 检查必要的配置段
    required_sections = ["llm_config", "feishu_config", "post_processing_config"]
    for section in required_sections:
        if section not in config:
            return False, f"配置缺少 '{section}' 部分"

    # 检查LLM配置
    llm_config = config["llm_config"]
    required_llm_keys = ["TARGET_COLUMNS", "BATCH_SIZE", "MAX_COMPLETION_TOKENS"]
    for key in required_llm_keys:
        if key not in llm_config:
            return False, f"LLM配置缺少 '{key}' 项"

    # 检查飞书配置
    feishu_config = config["feishu_config"]
    required_feishu_keys = ["APP_ID", "APP_SECRET", "APP_TOKEN", "TABLE_IDS"]
    for key in required_feishu_keys:
        if key not in feishu_config:
            return False, f"飞书配置缺少 '{key}' 项"

    # 检查类型
    if not isinstance(llm_config["TARGET_COLUMNS"], list):
        return False, "TARGET_COLUMNS 必须是列表类型"

    if not isinstance(feishu_config["TABLE_IDS"], list):
        return False, "TABLE_IDS 必须是列表类型"

    return True, "配置验证通过"
