import os
import shutil

# 读取文件
with open("processor.py", "r") as f:
    lines = f.readlines()

# 导入shutil模块（如果还没有）
has_shutil_import = False
for i, line in enumerate(lines):
    if "import shutil" in line:
        has_shutil_import = True
        break

if not has_shutil_import:
    for i, line in enumerate(lines):
        if line.startswith("import os"):
            lines[i] = "import os\nimport shutil\n"
            break

# 在文件上传成功后添加保存本地副本的代码
for i, line in enumerate(lines):
    if "✓ 文件上传成功" in line:
        copy_code = "        # 保留JSONL文件本地副本用于后续处理\n"
        copy_code += '        shutil.copy(jsonl_path, os.path.join(task_dir, "batch_input.jsonl"))\n'
        copy_code += "        print(f\"   ✓ 已保存JSONL文件本地副本: {os.path.join(task_dir, 'batch_input.jsonl')}\")\n"
        lines.insert(i + 1, copy_code)
        break

# 写回文件
with open("processor.py", "w") as f:
    f.writelines(lines)

print("成功修改 processor.py，添加了保存JSONL文件本地副本的代码")
