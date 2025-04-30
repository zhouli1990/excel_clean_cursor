#!/bin/bash
# 项目环境设置脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 确保Python正确安装
echo -e "${GREEN}检查Python版本...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python 3，请先安装Python 3.8或更高版本${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
echo -e "${GREEN}找到Python版本: ${PYTHON_VERSION}${NC}"

# 创建虚拟环境
echo -e "${GREEN}创建虚拟环境...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}发现已有的虚拟环境，是否重新创建? [y/N]${NC}"
    read -r recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}删除现有虚拟环境...${NC}"
        rm -rf .venv
        python3 -m venv .venv
        echo -e "${GREEN}已创建新的虚拟环境${NC}"
    else
        echo -e "${GREEN}使用现有虚拟环境${NC}"
    fi
else
    python3 -m venv .venv
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

# 安装依赖
echo -e "${GREEN}安装项目依赖...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
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