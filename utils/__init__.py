"""
工具模块包。提供以下组件：

- logger: 日志记录工具
- config_manager: 配置管理工具
- task_manager: 任务状态管理
- error_handler: 错误处理工具
- file_utils: 文件处理工具
"""

# 确保目录结构存在
import os

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)

# 导出模块主要组件
from utils.logger import setup_logger
from utils.config_manager import load_config, save_config, validate_config
from utils.task_manager import (
    get_task_info,
    create_task,
    update_task_progress,
    mark_task_completed,
    mark_task_failed,
    get_latest_task,
)
from utils.error_handler import (
    AppError,
    handle_exceptions,
    try_except_with_logging,
    format_error_response,
)
from utils.file_utils import (
    read_excel_or_csv,
    save_dataframe,
    get_file_info,
)
