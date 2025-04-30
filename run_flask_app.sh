#!/bin/bash

# 获取脚本所在目录
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$APP_DIR"

echo "正在启动表格数据清洗与飞书同步工具..."
echo "工作目录: $APP_DIR"

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "找到虚拟环境 .venv，正在激活..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "找到虚拟环境 venv，正在激活..."
    source venv/bin/activate
else
    echo "未找到虚拟环境，使用系统Python"
fi

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "错误: 未找到Python解释器"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# 启动Flask应用
echo "正在启动应用..."
exec $PYTHON_CMD app.py 