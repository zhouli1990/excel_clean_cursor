import traceback
import sys
import functools
import json
import uuid
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Type, Union

# 从logger模块导入日志工具
from utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger("error_handler", detailed_format=True)

# 定义错误类型与HTTP状态码映射
ERROR_TO_STATUS = {
    ValueError: 400,
    TypeError: 400,
    KeyError: 400,
    FileNotFoundError: 404,
    PermissionError: 403,
    ConnectionError: 503,
    TimeoutError: 504,
    Exception: 500,  # 默认状态码
}

# 错误日志目录
ERROR_LOG_DIR = "logs/errors"


class AppError(Exception):
    """
    应用程序自定义错误基类
    """

    def __init__(
        self, message: str, status_code: int = 500, details: Optional[Dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def log_error_with_context(
    error: Exception, context: Dict = None, log_traceback: bool = True
) -> str:
    """
    详细记录错误信息，包括上下文

    Args:
        error: 异常对象
        context: 额外的上下文信息
        log_traceback: 是否记录完整堆栈

    Returns:
        str: 错误ID
    """
    import os

    # 生成错误ID
    error_id = str(uuid.uuid4())[:8]

    # 准备错误信息
    error_info = {
        "error_id": error_id,
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
    }

    # 添加堆栈跟踪
    if log_traceback:
        error_info["traceback"] = traceback.format_exc()

    # 记录到标准logger
    logger.error(
        f"错误ID: {error_id} | 类型: {type(error).__name__} | 消息: {str(error)}",
        exc_info=log_traceback,
    )

    # 确保错误日志目录存在
    os.makedirs(ERROR_LOG_DIR, exist_ok=True)

    # 写入详细错误日志文件
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        error_log_file = f"{ERROR_LOG_DIR}/error_{date_str}.log"

        with open(error_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_info, ensure_ascii=False, indent=2))
            f.write("\n\n")
    except Exception as e:
        logger.error(f"写入错误日志文件失败: {e}")

    return error_id


def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    将异常格式化为统一的响应格式

    Args:
        error: 异常对象

    Returns:
        Dict: 格式化的错误响应
    """
    # 记录错误并获取错误ID
    error_id = log_error_with_context(error)

    if isinstance(error, AppError):
        return {
            "success": False,
            "error": error.message,
            "error_id": error_id,
            "status_code": error.status_code,
            "details": error.details,
        }

    # 根据异常类型确定状态码
    error_class = error.__class__
    status_code = ERROR_TO_STATUS.get(error_class, 500)

    return {
        "success": False,
        "error": str(error),
        "error_id": error_id,
        "status_code": status_code,
    }


def handle_exceptions(func: Callable) -> Callable:
    """
    异常处理装饰器，用于路由函数

    Args:
        func: 要装饰的函数

    Returns:
        Callable: 装饰后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 获取请求信息作为上下文
            context = {}
            try:
                from flask import request

                context = {
                    "url": request.url,
                    "method": request.method,
                    "headers": dict(request.headers),
                    "args": dict(request.args),
                    "remote_addr": request.remote_addr,
                    "endpoint": request.endpoint,
                }
                # 不记录文件数据，可能过大
                if request.is_json:
                    context["json"] = request.get_json()
            except Exception:
                pass

            # 记录详细错误
            error_id = log_error_with_context(e, context)

            # 格式化错误响应
            error_response = format_error_response(e)
            error_response["error_id"] = error_id

            # 从flask导入jsonify和make_response (在需要时导入避免循环导入)
            from flask import jsonify, make_response

            # 记录最终返回的错误响应
            logger.error(
                f"返回错误响应: ID={error_id}, 状态码={error_response['status_code']}, "
                f"消息={error_response.get('error', '无错误消息')}"
            )

            # 返回正确的HTTP状态码
            return make_response(jsonify(error_response), error_response["status_code"])

    return wrapper


def try_except_with_logging(
    default_value: Any = None,
    error_message: str = "操作失败",
    log_traceback: bool = True,
    raise_error: bool = False,
    expected_exceptions: Union[Type[Exception], tuple] = Exception,
) -> Callable:
    """
    通用的 try-except 装饰器，带日志记录

    Args:
        default_value: 出错时返回的默认值
        error_message: 错误消息前缀
        log_traceback: 是否记录完整的堆栈跟踪
        raise_error: 是否重新抛出异常
        expected_exceptions: 预期会捕获的异常类型

    Returns:
        Callable: 装饰后的函数
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module_name = func.__module__

            # 创建上下文信息
            context = {
                "function": func_name,
                "module": module_name,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            }

            try:
                return func(*args, **kwargs)
            except expected_exceptions as e:
                # 合并原有上下文和异常信息
                formatted_message = f"{error_message}: {str(e)}"

                # 记录错误
                error_id = log_error_with_context(e, context, log_traceback)

                logger.error(
                    f"函数 {module_name}.{func_name} 执行失败 [错误ID: {error_id}]: {formatted_message}"
                )

                if raise_error:
                    raise

                return default_value

        return wrapper

    return decorator
