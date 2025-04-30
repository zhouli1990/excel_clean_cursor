#!/bin/bash
# 项目环境设置脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 查找Python 3.11
echo -e "${GREEN}查找Python 3.11...${NC}"

# 尝试可能的Python 3.11命令位置
PYTHON_CMD=""
for cmd in "python3.11" "/usr/local/bin/python3.11" "/opt/homebrew/bin/python3.11"; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        break
    fi
done

# 如果没有找到Python 3.11
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}错误: 未找到Python 3.11${NC}"
    echo -e "${YELLOW}您可能需要安装Python 3.11:${NC}"
    echo -e "    brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo -e "${GREEN}找到Python版本: ${PYTHON_VERSION}${NC}"

# 创建虚拟环境
echo -e "${GREEN}创建虚拟环境...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}发现已有的虚拟环境，是否重新创建? [y/N]${NC}"
    read -r recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}删除现有虚拟环境...${NC}"
        rm -rf .venv
        $PYTHON_CMD -m venv .venv
        echo -e "${GREEN}已创建新的虚拟环境${NC}"
    else
        echo -e "${GREEN}使用现有虚拟环境${NC}"
    fi
else
    $PYTHON_CMD -m venv .venv
    echo -e "${GREEN}已创建虚拟环境${NC}"
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境...${NC}"
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS或Linux
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    echo -e "${RED}无法确定操作系统类型，请手动激活虚拟环境${NC}"
    exit 1
fi

# 更新pip
echo -e "${GREEN}更新pip...${NC}"
python -m pip install --upgrade pip

# 先安装setuptools
echo -e "${GREEN}安装基础工具...${NC}"
pip install setuptools wheel

# 安装依赖
echo -e "${GREEN}安装项目依赖...${NC}"
if [ -f "requirements.txt" ]; then
    # 尝试安装依赖
    if ! pip install -r requirements.txt; then
        echo -e "${YELLOW}标准安装失败，尝试逐个安装核心依赖...${NC}"
        pip install flask==2.2.3
        pip install requests==2.28.2
        pip install openpyxl==3.1.2
        pip install pandas numpy
    fi
    echo -e "${GREEN}已安装项目依赖${NC}"
else
    echo -e "${RED}未找到requirements.txt文件${NC}"
    exit 1
fi

# 检查依赖安装情况
echo -e "${GREEN}验证依赖安装...${NC}"
python check_dependencies.py

# 设置完成
echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}项目设置完成!${NC}"
echo -e "${GREEN}使用以下命令激活虚拟环境:${NC}"
echo -e "    ${YELLOW}source .venv/bin/activate${NC}  # macOS/Linux"
echo -e "    ${YELLOW}.venv\\Scripts\\activate${NC}      # Windows"
echo -e "${GREEN}运行项目:${NC}"
echo -e "    ${YELLOW}python app.py${NC}"
echo -e "${GREEN}=====================================${NC}" 