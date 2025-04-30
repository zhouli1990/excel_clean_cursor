#!/usr/bin/env python3
"""
检查项目依赖状态
- 检查requirements.txt中列出的包是否已安装
- 比较已安装版本与requirements.txt中指定的版本
- 提供安装命令建议
"""
import sys
import subprocess
import re
import os
from typing import Dict, List, Tuple


def parse_requirements(file_path: str) -> Dict[str, str]:
    """解析requirements.txt文件"""
    if not os.path.exists(file_path):
        print(f"错误: 找不到依赖文件 {file_path}")
        return {}

    packages = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith("#"):
                continue

            # 解析包名和版本
            if "==" in line:
                name, version = line.split("==", 1)
                packages[name.strip().lower()] = version.strip()

    return packages


def get_installed_packages() -> Dict[str, str]:
    """获取当前环境中已安装的包"""
    packages = {}
    try:
        # 使用pip list获取已安装的包
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )

        # 解析JSON输出
        import json

        try:
            pkg_list = json.loads(result.stdout)
            for pkg_info in pkg_list:
                name = pkg_info["name"].lower()
                version = pkg_info["version"]
                packages[name] = version
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
                    name, version = line.split("==", 1)
                    packages[name.lower()] = version
    except subprocess.CalledProcessError as e:
        print(f"运行pip命令时出错: {e}")

    return packages


def compare_dependencies() -> Tuple[List[str], List[str], List[str]]:
    """比较requirements.txt和已安装的包"""
    req_packages = parse_requirements("requirements.txt")
    installed_packages = get_installed_packages()

    # 比较结果
    missing = []  # 未安装的包
    version_mismatch = []  # 版本不匹配的包
    ok = []  # 已安装且版本匹配的包

    for pkg_name, req_version in req_packages.items():
        if pkg_name not in installed_packages:
            missing.append(f"{pkg_name}=={req_version}")
        else:
            inst_version = installed_packages[pkg_name]
            if inst_version != req_version:
                version_mismatch.append(
                    f"{pkg_name}: 需要 {req_version}，已安装 {inst_version}"
                )
            else:
                ok.append(f"{pkg_name}=={req_version}")

    return missing, version_mismatch, ok


def print_report():
    """打印依赖检查报告"""
    missing, version_mismatch, ok = compare_dependencies()

    print("\n===== 项目依赖检查报告 =====")

    # 打印已安装且版本匹配的包
    print(f"\n✅ 已安装且版本匹配的包: {len(ok)}")
    for pkg in sorted(ok):
        print(f"  {pkg}")

    # 打印版本不匹配的包
    if version_mismatch:
        print(f"\n⚠️ 版本不匹配的包: {len(version_mismatch)}")
        for pkg in sorted(version_mismatch):
            print(f"  {pkg}")

    # 打印未安装的包
    if missing:
        print(f"\n❌ 未安装的包: {len(missing)}")
        for pkg in sorted(missing):
            print(f"  {pkg}")

        # 生成安装命令
        cmd = f"{sys.executable} -m pip install " + " ".join(missing)
        print(f"\n📦 安装命令:")
        print(f"  {cmd}")

    # 总结
    if not missing and not version_mismatch:
        print("\n🎉 所有依赖已正确安装！")
    else:
        print(
            f"\n📋 总结: {len(ok)} 个已安装，{len(version_mismatch)} 个版本不匹配，{len(missing)} 个未安装。"
        )


if __name__ == "__main__":
    print_report()
