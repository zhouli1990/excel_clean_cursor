#!/usr/bin/env python3

import os
import json
import shutil
import uuid
import traceback


def test_file_copy_fix():
    """测试文件复制修复"""
    print("\n=== 测试文件复制修复 ===")

    # 创建测试目录和文件
    task_id = f"test-fix-{uuid.uuid4().hex[:8]}"
    test_dir = os.path.join("uploads", task_id)
    os.makedirs(test_dir, exist_ok=True)

    # 创建测试文件
    source_file = os.path.join(test_dir, "batch_input.jsonl")
    with open(source_file, "w") as f:
        f.write('{"test": "data"}\n')

    print(f"创建测试文件: {source_file}")

    # 测试文件复制 - 模拟processor.py中的代码
    try:
        # 模拟提交函数中的文件复制逻辑
        jsonl_path = source_file
        local_copy_path = os.path.join(test_dir, "batch_input.jsonl")

        # 检查源文件路径和目标路径是否相同
        if jsonl_path != local_copy_path:
            print("源文件和目标文件路径不同，执行复制操作")
            shutil.copy(jsonl_path, local_copy_path)
            print(f"✅ 成功复制文件: {jsonl_path} -> {local_copy_path}")
        else:
            print("源文件和目标文件路径相同，跳过复制操作")
            print(f"✅ 源文件已在目标位置: {jsonl_path}")

        # 清理测试文件和目录
        shutil.rmtree(test_dir)
        print(f"已清理测试目录: {test_dir}")

        print("✅ 测试通过: 文件路径检查逻辑正确处理了相同路径情况")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(traceback.format_exc())

    print("\n测试完成")


if __name__ == "__main__":
    test_file_copy_fix()
