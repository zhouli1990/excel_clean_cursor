import traceback
import sys
import functools
from typing import Callable, Dict, Any, Optional, Type, Union

# 从logger模块导入日志工具
from utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger("error_handler")

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


def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    将异常格式化为统一的响应格式

    Args:
        error: 异常对象

    Returns:
        Dict: 格式化的错误响应
    """
    if isinstance(error, AppError):
        return {
            "success": False,
            "error": error.message,
            "status_code": error.status_code,
            "details": error.details,
        }

    # 根据异常类型确定状态码
    error_class = error.__class__
    status_code = ERROR_TO_STATUS.get(error_class, 500)

    return {"success": False, "error": str(error), "status_code": status_code}


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
            error_type = type(e).__name__
            error_message = str(e)

            # 记录详细错误
            logger.error(f"{error_type}: {error_message}")
            logger.debug(traceback.format_exc())

            # 格式化错误响应
            error_response = format_error_response(e)

            # 从flask导入jsonify和make_response (在需要时导入避免循环导入)
            from flask import jsonify, make_response

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
            try:
                return func(*args, **kwargs)
            except expected_exceptions as e:
                formatted_message = f"{error_message}: {str(e)}"
                logger.error(formatted_message)

                if log_traceback:
                    logger.debug(traceback.format_exc())

                if raise_error:
                    raise

                return default_value

        return wrapper

    return decorator
