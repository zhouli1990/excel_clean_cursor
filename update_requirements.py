#!/usr/bin/env python3
"""
自动更新requirements.txt文件
- 检查项目中使用的关键依赖
- 记录当前环境中包的精确版本
- 排除开发依赖
"""
import os
import sys
import subprocess
import re
from datetime import datetime

# 项目核心依赖包列表 - 根据项目特性添加或删除
CORE_DEPENDENCIES = [
    "flask",  # Web框架
    "pandas",  # 数据处理
    "numpy",  # 数值计算
    "openpyxl",  # Excel处理
    "requests",  # HTTP请求
    "werkzeug",  # Flask依赖
    "jinja2",  # 模板引擎
    "itsdangerous",  # 安全签名
    "blinker",  # 信号支持
    "click",  # 命令行接口
    "pytz",  # 时区处理
]

# 排除的开发工具包
EXCLUDED_PACKAGES = [
    "ipykernel",
    "jupyter",
    "pytest",
    "pylint",
    "black",
    "mypy",
    "flake8",
    "pip",
    "setuptools",
    "wheel",
    "twine",
    "tox",
    "coverage",
    "build",
    "autopep8",
    "isort",
]


def scan_imports_from_file(file_path):
    """从文件中提取导入的包名"""
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # 正则匹配import语句
            import_pattern = re.compile(r"^import\s+([a-zA-Z0-9_,.]+)", re.MULTILINE)
            from_pattern = re.compile(
                r"^from\s+([a-zA-Z0-9_.]+)\s+import", re.MULTILINE
            )

            content = f.read()
            # 查找所有import语句
            for match in import_pattern.finditer(content):
                # 处理多个导入，如import os, sys, re
                for pkg in match.group(1).split(","):
                    # 只取主包名，如os.path只取os
                    pkg_name = pkg.strip().split(".")[0]
                    if pkg_name:
                        imports.add(pkg_name)

            # 查找所有from...import语句
            for match in from_pattern.finditer(content):
                # 只取主包名，如from os.path import join只取os
                pkg_name = match.group(1).split(".")[0]
                if pkg_name:
                    imports.add(pkg_name)
    except (UnicodeDecodeError, IOError) as e:
        print(f"警告: 无法解析文件 {file_path}: {e}")

    return imports


def get_project_imports():
    """扫描项目文件获取所有导入包"""
    all_imports = set()

    # 扫描所有.py文件
    for root, _, files in os.walk("."):
        # 排除虚拟环境和缓存目录
        if "/.venv/" in root or "/__pycache__/" in root or "/." in root:
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports = scan_imports_from_file(file_path)
                all_imports.update(imports)

    return all_imports


def get_pip_freeze_packages():
    """获取当前环境中安装的所有包"""
    try:
        # 使用pip list代替pip freeze，格式为：Package Version
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )

        # 解析JSON输出
        import json

        packages = {}
        try:
            pkg_list = json.loads(result.stdout)
            for pkg_info in pkg_list:
                name = pkg_info["name"].lower()
                version = pkg_info["version"]
                packages[name] = f"{name}=={version}"
        except json.JSONDecodeError:
            print("警告: 无法解析pip list的JSON输出，尝试使用pip freeze")
            # 回退到pip freeze
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                if "==" in line:
                    name = line.split("==")[0].lower()
                    packages[name] = line

        print(f"当前环境中找到 {len(packages)} 个已安装的包")
        return packages
    except subprocess.CalledProcessError as e:
        print(f"运行pip命令时出错: {e}")
        return {}


def update_requirements_file():
    """更新requirements.txt文件"""
    print("开始更新项目依赖...")

    # 获取pip freeze的所有包
    print("获取当前环境中的包...")
    all_packages = get_pip_freeze_packages()

    # 合并扫描和手动指定的依赖
    print("扫描项目文件中的import语句...")
    imported_packages = get_project_imports()

    # 筛选出有效的第三方依赖
    requirements = []

    # 先添加核心依赖
    print("\n添加核心依赖:")
    for pkg in CORE_DEPENDENCIES:
        pkg_lower = pkg.lower()
        if pkg_lower in all_packages:
            requirements.append(all_packages[pkg_lower])
            print(f"  ✓ {all_packages[pkg_lower]}")
        else:
            print(f"  ✗ {pkg} (未安装)")

    # 添加项目中导入但不在核心依赖中的包
    print("\n添加其他项目导入的包:")
    for pkg in imported_packages:
        pkg_lower = pkg.lower()
        # 排除标准库和开发工具
        if (
            pkg_lower not in [dep.lower() for dep in CORE_DEPENDENCIES]
            and pkg_lower in all_packages
            and not any(exclude in pkg_lower for exclude in EXCLUDED_PACKAGES)
        ):
            # 检查是否已在requirements中
            if not any(
                req.lower().startswith(pkg_lower + "==") for req in requirements
            ):
                requirements.append(all_packages[pkg_lower])
                print(f"  + {all_packages[pkg_lower]}")

    # 确保requirements不为空
    if not requirements:
        print("\n警告: 未找到有效依赖，添加基本依赖...")
        # 添加基本依赖
        for pkg in ["flask", "pandas", "openpyxl"]:
            if pkg in all_packages:
                requirements.append(all_packages[pkg])
                print(f"  + {all_packages[pkg]} (基础依赖)")

    # 排序
    requirements.sort()

    # 备份旧的requirements.txt
    if os.path.exists("requirements.txt"):
        try:
            with open("requirements.txt", "r", encoding="utf-8") as f:
                old_content = f.read()
            backup_file = (
                f'requirements.txt.bak.{datetime.now().strftime("%Y%m%d%H%M%S")}'
            )
            with open(backup_file, "w", encoding="utf-8") as f:
                f.write(old_content)
            print(f"\n已备份旧的依赖文件到 {backup_file}")
        except Exception as e:
            print(f"备份旧的依赖文件时出错: {e}")

    # 添加头部注释
    header = f"# 自动更新于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += "# 此文件由update_requirements.py自动生成\n\n"

    # 写入新的requirements.txt
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(requirements) + "\n")

    print(f"\n已更新requirements.txt，包含 {len(requirements)} 个依赖")

    return True


if __name__ == "__main__":
    update_requirements_file()
