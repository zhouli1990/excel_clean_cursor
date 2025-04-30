import os
import ast
import re
from datetime import datetime

# --- 配置 ---
PROJECT_NAME = "表格数据清洗与飞书同步工具"
OUTPUT_FILE = "项目结构.md"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 要排除的目录和文件模式 (正则表达式)
EXCLUDE_DIRS_PATTERN = r"(__pycache__|venv|env|\.venv|\.env|\.git|\.idea|\.vscode|\.cursor|node_modules|build|dist|uploads|outputs)"
EXCLUDE_FILES_PATTERN = r"(\.DS_Store|Thumbs\.db|\.gitignore|generate_project_structure\.py|项目结构\.md|.*\.log|.*\.tmp)"
# ---


def should_exclude(path, is_dir):
    """检查路径是否应该被排除"""
    path_parts = path.split(os.sep)
    if any(re.match(EXCLUDE_DIRS_PATTERN, part) for part in path_parts):
        return True
    if not is_dir and re.match(EXCLUDE_FILES_PATTERN, os.path.basename(path)):
        return True
    return False


def generate_file_tree(start_path):
    """生成项目文件树结构"""
    # 使用系统命令获取更准确的目录树结构
    try:
        import subprocess

        # 尝试使用find命令获取所有文件和目录
        cmd = f"find {start_path} -type f -o -type d | sort"
        result = subprocess.check_output(cmd, shell=True, text=True).strip().split("\n")

        # 过滤掉应该排除的路径
        filtered_paths = []
        for path in result:
            is_dir = os.path.isdir(path)
            if not should_exclude(path, is_dir):
                filtered_paths.append((path, is_dir))

        # 构建树形结构
        tree_lines = []
        root_name = os.path.basename(start_path)
        tree_lines.append(root_name)

        for path, is_dir in filtered_paths:
            # 跳过根目录自身
            if path == start_path:
                continue

            # 获取相对路径
            rel_path = os.path.relpath(path, start_path)
            if rel_path == ".":
                continue

            # 计算深度和前缀
            parts = rel_path.split(os.sep)
            depth = len(parts) - 1

            # 生成正确的树形结构前缀
            prefix = ""
            for i in range(depth):
                prefix += "│   "

            # 处理目录和文件
            name = os.path.basename(path)
            if is_dir:
                tree_lines.append(f"{prefix}├── {name}/")
            else:
                tree_lines.append(f"{prefix}├── {name}")

        return "\n".join(tree_lines)

    except (subprocess.SubprocessError, Exception) as e:
        print(f"使用系统命令生成目录树失败: {e}")

        # 回退到纯Python实现
        tree_lines = []
        tree_lines.append(os.path.basename(start_path))

        for root, dirs, files in os.walk(start_path):
            # 过滤排除的目录
            dirs[:] = [
                d for d in dirs if not should_exclude(os.path.join(root, d), True)
            ]
            # 过滤排除的文件
            files = [
                f for f in files if not should_exclude(os.path.join(root, f), False)
            ]

            # 跳过根目录
            if root == start_path:
                continue

            # 计算相对路径和深度
            rel_path = os.path.relpath(root, start_path)
            depth = len(rel_path.split(os.sep))

            # 添加目录
            prefix = "│   " * (depth - 1)
            tree_lines.append(f"{prefix}├── {os.path.basename(root)}/")

            # 添加文件
            file_prefix = "│   " * depth
            for f in sorted(files):
                tree_lines.append(f"{file_prefix}├── {f}")

        return "\n".join(tree_lines)


def parse_python_file(filepath):
    """解析Python文件，提取类、函数和Flask路由信息"""
    routes = []
    classes = []
    functions = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
            tree = ast.parse(source, filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node) or "无文档字符串"
                    # 改进处理多行文档字符串，保留更多有用信息
                    if "\n\n" in docstring:  # 有段落分隔
                        doc_summary = docstring.split("\n\n")[0].replace("\n", " ")
                    else:  # 没有段落分隔
                        lines = docstring.split("\n")
                        doc_summary = lines[0]  # 默认取第一行
                        if (
                            len(lines) > 1
                            and lines[1].strip()
                            and not lines[1].strip().startswith("Path")
                            and not lines[1].strip().startswith("Request")
                        ):
                            # 如果第二行不是参数说明，合并为更详细的摘要
                            doc_summary += " " + lines[1].strip()

                    # 限制摘要长度
                    if len(doc_summary) > 100:
                        doc_summary = doc_summary[:97] + "..."

                    classes.append({"name": node.name, "docstring": doc_summary})
                elif isinstance(node, ast.FunctionDef):
                    # 检查是否为Flask路由
                    is_route = False
                    route_path = None
                    methods = ["GET"]  # 默认方法
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            func_name = ""
                            if isinstance(
                                decorator.func, ast.Attribute
                            ):  # @app.route(...)
                                if decorator.func.attr == "route":
                                    func_name = "route"
                            elif isinstance(decorator.func, ast.Name):  # @route(...)
                                func_name = decorator.func.id

                            if func_name == "route":
                                is_route = True
                                # 提取路径
                                if decorator.args and isinstance(
                                    decorator.args[0], ast.Constant
                                ):
                                    route_path = decorator.args[0].value
                                # 提取方法
                                for keyword in decorator.keywords:
                                    if keyword.arg == "methods" and isinstance(
                                        keyword.value, (ast.List, ast.Tuple)
                                    ):
                                        methods = [
                                            m.value
                                            for m in keyword.value.elts
                                            if isinstance(m, ast.Constant)
                                        ]
                                break  # 找到route装饰器就停止

                    # 提取文档字符串，以更友好的方式处理
                    docstring = ast.get_docstring(node) or "无文档字符串"

                    # 尝试提取更有意义的描述
                    if "\n" in docstring:
                        # 查找第一段落的结束
                        lines = docstring.split("\n")

                        # 跳过空行
                        first_line = next(
                            (line for line in lines if line.strip()), "无描述"
                        )

                        # 查找第一个空行之前的所有内容作为描述
                        description_lines = []
                        for line in lines:
                            if not line.strip():  # 遇到空行停止
                                break
                            if line.strip().startswith(
                                ("Request:", "Path Params:", "Returns:", "Requires:")
                            ):
                                # 遇到参数说明停止
                                break
                            description_lines.append(line.strip())

                        # 组合描述行
                        if description_lines:
                            doc_summary = " ".join(description_lines)
                        else:
                            doc_summary = first_line
                    else:
                        # 简单文档，直接使用
                        doc_summary = docstring.strip()

                    # 截断过长的描述
                    if len(doc_summary) > 150:
                        doc_summary = doc_summary[:147] + "..."

                    # 为路由函数提取请求和返回信息
                    if is_route:
                        # 尝试提取请求参数信息
                        request_info = ""
                        if "Request:" in docstring:
                            request_section = docstring.split("Request:")[1].split(
                                "\n\n"
                            )[0]
                            request_lines = [
                                line.strip()
                                for line in request_section.split("\n")
                                if line.strip()
                                and not line.strip().startswith("Returns:")
                            ]
                            if request_lines:
                                request_info = "; ".join(request_lines)

                        # 尝试提取返回信息
                        return_info = ""
                        if "Returns:" in docstring:
                            return_section = docstring.split("Returns:")[1].split(
                                "\n\n"
                            )[0]
                            return_lines = [
                                line.strip()
                                for line in return_section.split("\n")
                                if line.strip()
                            ]
                            if return_lines:
                                return_info = "; ".join(return_lines)

                        routes.append(
                            {
                                "path": route_path or "未知路径",
                                "methods": ", ".join(methods),
                                "function": node.name,
                                "docstring": doc_summary,
                                "request_info": request_info,
                                "return_info": return_info,
                            }
                        )
                    else:
                        # 只添加非内部函数 (非下划线开头) 或特殊方法(__init__)
                        if not node.name.startswith("_") or node.name == "__init__":
                            functions.append(
                                {"name": node.name, "docstring": doc_summary}
                            )

    except Exception as e:
        print(f"无法解析文件 {filepath}: {e}")
    return routes, classes, functions


def find_frontend_assets(start_path):
    """查找JS和CSS文件"""
    js_files = []
    css_files = []
    for root, _, files in os.walk(start_path):
        if should_exclude(root, True):
            continue
        for file in files:
            filepath = os.path.join(root, file)
            if should_exclude(filepath, False):
                continue
            rel_path = os.path.relpath(filepath, ROOT_DIR)
            if file.endswith(".js"):
                js_files.append(rel_path)
            elif file.endswith(".css"):
                css_files.append(rel_path)
    return sorted(js_files), sorted(css_files)


def generate_structure_markdown():
    """生成完整的项目结构Markdown文档"""
    markdown = []
    markdown.append(f"# {PROJECT_NAME} 项目结构")
    markdown.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    markdown.append("")

    # --- 1. 总体目录结构 ---
    markdown.append("## 1. 总体目录结构")
    markdown.append("")
    markdown.append("```")
    # 在这里调用 file tree 生成函数
    tree_structure = generate_file_tree(ROOT_DIR)
    markdown.append(tree_structure)
    markdown.append("```")
    markdown.append("")

    # --- 2. 核心模块与功能详解 ---
    markdown.append("## 2. 核心模块与功能详解")
    markdown.append("")

    all_routes = []
    python_details = {}  # file -> {'classes': [], 'functions': []}

    for root, dirs, files in os.walk(ROOT_DIR):
        # 过滤排除的目录
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), True)]

        for file in sorted(files):
            filepath = os.path.join(root, file)
            if should_exclude(filepath, False):
                continue

            rel_path = os.path.relpath(filepath, ROOT_DIR)

            if file.endswith(".py"):
                routes, classes, functions = parse_python_file(filepath)
                if routes or classes or functions:
                    python_details[rel_path] = {
                        "classes": classes,
                        "functions": functions,
                    }
                all_routes.extend(routes)  # 收集所有路由

    # --- 2.1 主要API路由 (基于Flask @app.route) ---
    if all_routes:
        markdown.append("### 2.1 主要API路由 (Flask)")
        markdown.append("从 `.py` 文件中 `@app.route` 或类似装饰器提取的接口：")
        markdown.append("")

        # 首先按HTTP方法分组
        get_routes = [r for r in all_routes if "GET" in r["methods"]]
        post_routes = [r for r in all_routes if "POST" in r["methods"]]
        other_routes = [
            r
            for r in all_routes
            if "GET" not in r["methods"] and "POST" not in r["methods"]
        ]

        # GET接口表格
        if get_routes:
            markdown.append("#### GET接口")
            markdown.append("| 路由路径 | 处理函数 | 功能描述 |")
            markdown.append("|----------|----------|----------|")
            for route in sorted(get_routes, key=lambda x: x["path"]):
                # 查找对应的函数定义文件
                route_file = "未知文件"
                for file, details in python_details.items():
                    if any(
                        func["name"] == route["function"]
                        for func in details["functions"]
                    ):
                        route_file = file
                        break

                func_doc = route["docstring"]
                markdown.append(
                    f"| `{route['path']}` | `{route_file}::{route['function']}()` | {func_doc} |"
                )
            markdown.append("")

        # POST接口表格
        if post_routes:
            markdown.append("#### POST接口")
            markdown.append("| 路由路径 | 处理函数 | 功能描述 |")
            markdown.append("|----------|----------|----------|")
            for route in sorted(post_routes, key=lambda x: x["path"]):
                # 查找对应的函数定义文件
                route_file = "未知文件"
                for file, details in python_details.items():
                    if any(
                        func["name"] == route["function"]
                        for func in details["functions"]
                    ):
                        route_file = file
                        break

                func_doc = route["docstring"]
                markdown.append(
                    f"| `{route['path']}` | `{route_file}::{route['function']}()` | {func_doc} |"
                )
            markdown.append("")

        # 其他方法接口表格
        if other_routes:
            markdown.append("#### 其他HTTP方法接口")
            markdown.append("| 路由路径 | HTTP方法 | 处理函数 | 功能描述 |")
            markdown.append("|----------|----------|----------|----------|")
            for route in sorted(other_routes, key=lambda x: x["path"]):
                # 查找对应的函数定义文件
                route_file = "未知文件"
                for file, details in python_details.items():
                    if any(
                        func["name"] == route["function"]
                        for func in details["functions"]
                    ):
                        route_file = file
                        break

                func_doc = route["docstring"]
                markdown.append(
                    f"| `{route['path']}` | `{route['methods']}` | `{route_file}::{route['function']}()` | {func_doc} |"
                )
            markdown.append("")

        # 添加API详细描述部分
        markdown.append("### 2.1.1 API接口详细说明")
        markdown.append("")
        for route in sorted(all_routes, key=lambda x: x["path"]):
            markdown.append(f"#### `{route['methods']} {route['path']}`")
            markdown.append(f"- **处理函数**: `{route['function']}()`")
            markdown.append(f"- **功能描述**: {route['docstring']}")

            if "request_info" in route and route["request_info"]:
                markdown.append(f"- **请求参数**: {route['request_info']}")

            if "return_info" in route and route["return_info"]:
                markdown.append(f"- **返回值**: {route['return_info']}")

            markdown.append("")

    # --- 2.2 Python 模块详情 ---
    markdown.append("## 2.2 Python 模块详情 (`.py`)")
    markdown.append("按文件列出主要的类和函数：")
    markdown.append("")

    if not python_details:
        markdown.append("未找到或解析任何 Python 文件中的类或函数。")
    else:
        for filepath, details in sorted(python_details.items()):
            if not details["classes"] and not details["functions"]:
                continue  # Skip files with no parsed elements

            markdown.append(f"#### ` {filepath} `")
            markdown.append("")
            if details["classes"]:
                markdown.append("- **类:**")
                for cls in sorted(details["classes"], key=lambda x: x["name"]):
                    markdown.append(f"  - `{cls['name']}`: {cls['docstring']}")
                markdown.append("")  # Add spacing if there are also functions
            if details["functions"]:
                markdown.append("- **函数:**")
                for func in sorted(details["functions"], key=lambda x: x["name"]):
                    # Skip route functions if already listed in routes table? Maybe not, show all.
                    markdown.append(f"  - `{func['name']}()`: {func['docstring']}")
            markdown.append("---")  # Separator between files
    markdown.append("")

    # --- 3. 前端资源 ---
    markdown.append("## 3. 前端资源")
    markdown.append("")
    js_files, css_files = find_frontend_assets(ROOT_DIR)

    if js_files:
        markdown.append("### 3.1 JavaScript 文件 (`.js`)")
        for js_file in js_files:
            markdown.append(f"- `{js_file}`")
        markdown.append("")
    else:
        markdown.append("未找到 JavaScript 文件。")
        markdown.append("")

    if css_files:
        markdown.append("### 3.2 CSS 文件 (`.css`)")
        for css_file in css_files:
            markdown.append(f"- `{css_file}`")
        markdown.append("")
    else:
        markdown.append("未找到 CSS 文件。")
        markdown.append("")

    # --- 4. 数据处理流程概述 (可选，可以从之前的脚本或文档中获取) ---
    markdown.append("## 4. 数据处理流程概述")
    markdown.append("")
    markdown.append("```mermaid")
    markdown.append("graph TD")
    markdown.append("    A[用户上传 Excel/CSV] --> B{Web服务器接收文件};")
    markdown.append("    B --> C{后台任务启动};")
    markdown.append("    C --> D[processor.py 调用LLM];")
    markdown.append("    D --> E{获取LLM处理结果};")
    markdown.append("    C --> F[feishu_utils.py 拉取飞书数据];")
    markdown.append("    F --> G{获取飞书现有数据};")
    markdown.append("    E & G --> H[数据合并与初步处理];")
    markdown.append("    H --> I[postprocessor.py 后处理校验];")
    markdown.append("    I --> J{生成待审核结果文件};")
    markdown.append("    J --> K[用户下载审核];")
    markdown.append("    K --> L{用户上传修改后文件};")
    markdown.append("    L --> M[diff_utils.py 比较差异];")
    markdown.append("    M --> N{生成最终同步数据};")
    markdown.append("    N --> O[feishu_utils.py 同步到飞书];")
    markdown.append("    O --> P[完成];")
    markdown.append("```")
    markdown.append("(这是一个简化的流程图，具体细节参考代码)")
    markdown.append("")

    # --- 写入文件 ---
    try:
        with open(os.path.join(ROOT_DIR, OUTPUT_FILE), "w", encoding="utf-8") as f:
            f.write("\n".join(markdown))
        print(f"项目结构图已生成/更新到: {OUTPUT_FILE}")
    except IOError as e:
        print(f"错误：无法写入文件 {OUTPUT_FILE}: {e}")


if __name__ == "__main__":
    generate_structure_markdown()
