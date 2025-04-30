# 表格数据清洗与飞书同步工具

## 项目依赖管理

本项目提供了几个脚本用于管理项目依赖：

### 1. 环境设置

运行以下命令设置开发环境（创建虚拟环境并安装依赖）：

```bash
./setup.sh
```

该脚本会：
- 检查Python版本
- 创建虚拟环境（.venv）
- 安装requirements.txt中的依赖
- 验证依赖安装状态

### 2. 依赖检查

运行以下命令检查项目依赖的安装状态：

```bash
./check_dependencies.py
```

该脚本会：
- 检查requirements.txt中列出的依赖是否已安装
- 比较已安装版本与requirements.txt中指定的版本
- 如有必要，提供安装命令

### 3. 更新依赖清单

更新requirements.txt文件：

```bash
./update_requirements.py
```

该脚本会：
- 扫描项目文件中的import语句
- 提取非标准库的依赖包
- 记录当前环境中包的精确版本
- 备份旧的requirements.txt文件
- 更新requirements.txt文件

### 4. 依赖安装

手动安装项目依赖：

```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 部署说明

本项目适用于私有化部署到macOS系统：

1. 克隆项目代码到本地
2. 运行`./setup.sh`设置环境
3. 使用`python app.py`启动应用
4. 通过`http://localhost:5100`访问应用界面

## 依赖版本说明

项目依赖固定版本以确保一致性。核心依赖包括：

- Flask 2.2.3：Web框架
- Pandas 1.5.3：数据处理
- NumPy 1.24.2：数值计算
- openpyxl 3.1.2：Excel处理
- Requests 2.28.2：HTTP请求 