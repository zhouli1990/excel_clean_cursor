#!/usr/bin/env python3

import os
import pandas as pd
import uuid
import json
import traceback
import shutil
import time
from processor import process_files_and_consolidate


def create_test_files():
    """创建测试文件"""
    print("\n=== 创建测试文件 ===")

    # 确保测试目录存在
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)

    # 创建第一个测试文件
    test_data1 = {
        "公司名称": ["测试公司A1", "测试公司A2", "测试公司A3"],
        "联系人": ["张三", "李四", "王五"],
        "职位": ["经理", "总监", "工程师"],
        "电话": ["13800138001", "13900139002", "13700137003"],
        "local_row_id": [str(uuid.uuid4()) for _ in range(3)],
    }
    df1 = pd.DataFrame(test_data1)
    file1_path = os.path.join(test_dir, "test_file1.xlsx")
    df1.to_excel(file1_path, index=False)
    print(f"✅ 已创建测试文件1: {file1_path} (3行数据)")

    # 创建第二个测试文件
    test_data2 = {
        "公司名称": ["测试公司B1", "测试公司B2"],
        "联系人": ["赵六", "钱七"],
        "职位": ["销售", "客服"],
        "电话": ["13600136004", "13500135005"],
        "local_row_id": [str(uuid.uuid4()) for _ in range(2)],
    }
    df2 = pd.DataFrame(test_data2)
    file2_path = os.path.join(test_dir, "test_file2.xlsx")
    df2.to_excel(file2_path, index=False)
    print(f"✅ 已创建测试文件2: {file2_path} (2行数据)")

    return [file1_path, file2_path]


def test_multi_file_process():
    """测试多文件处理流程"""
    try:
        print("\n=== 测试多文件批处理功能 ===")

        # 创建测试文件
        input_files = create_test_files()

        # 设置测试输出路径
        output_path = "test_data/output.xlsx"

        # 测试配置
        config = {
            "TARGET_COLUMNS": ["公司名称", "联系人", "职位", "电话", "来源"],
            "DASHSCOPE_API_KEY": "sk-fake-test-key",  # 假API Key，测试会在实际API调用前失败
            "BAILIAN_MODEL_NAME": "qwen-plus",
            "BAILIAN_COMPLETION_WINDOW": "24h",
            "BATCH_SIZE": 2,  # 小批量以验证批次划分逻辑
        }

        # 进度回调函数
        def progress_callback(status, progress, files_done, total_files):
            print(f"[进度] {status} - {progress}% ({files_done}/{total_files})")

        # 调用处理函数
        print("\n>> 开始处理多文件...")
        try:
            result = process_files_and_consolidate(
                input_files, output_path, config, progress_callback
            )
            print(f"✅ 处理函数执行完成: {result}")
        except Exception as e:
            if "API Key" in str(e) or "sk-fake-test-key" in str(e):
                print(f"✅ 预期的API错误: {e}")
                # API错误是预期行为，因为我们使用了假API Key
                # 检查是否创建了预期的中间文件
                task_id = None

                # 查找最新的task目录
                uploads_dir = "uploads"
                if os.path.exists(uploads_dir):
                    task_dirs = [
                        d
                        for d in os.listdir(uploads_dir)
                        if os.path.isdir(os.path.join(uploads_dir, d))
                    ]
                    if task_dirs:
                        task_dirs.sort(
                            key=lambda x: os.path.getmtime(
                                os.path.join(uploads_dir, x)
                            ),
                            reverse=True,
                        )
                        task_id = task_dirs[0]

                if task_id:
                    print(f"\n>> 检查任务目录: uploads/{task_id}")
                    task_dir = os.path.join(uploads_dir, task_id)

                    # 检查批次输入文件
                    jsonl_path = os.path.join(task_dir, "batch_input.jsonl")
                    if os.path.exists(jsonl_path):
                        print(f"✅ 找到批次输入文件: {jsonl_path}")
                        with open(jsonl_path, "r", encoding="utf-8") as f:
                            batch_lines = f.readlines()
                            print(f"   - 包含 {len(batch_lines)} 个批次请求")

                            # 检查每个批次是否包含文件索引前缀
                            file_prefixes = set()
                            for i, line in enumerate(batch_lines):
                                try:
                                    record = json.loads(line)
                                    custom_id = record.get("custom_id", "")
                                    if (
                                        custom_id.startswith("file")
                                        and "_batch_" in custom_id
                                    ):
                                        prefix = custom_id.split("_batch_")[0]
                                        file_prefixes.add(prefix)
                                        print(f"   - 批次 {i+1}: {custom_id}")
                                except Exception as e:
                                    print(f"   ❌ 解析批次 {i+1} 失败: {e}")

                            print(
                                f"   - 发现 {len(file_prefixes)} 个文件前缀: {', '.join(sorted(file_prefixes))}"
                            )
                    else:
                        print(f"❌ 未找到批次输入文件: {jsonl_path}")

                    # 检查批次映射文件
                    mapping_path = os.path.join(task_dir, "batch_mapping.json")
                    if os.path.exists(mapping_path):
                        print(f"✅ 找到批次映射文件: {mapping_path}")
                        with open(mapping_path, "r", encoding="utf-8") as f:
                            mapping_data = json.load(f)
                            if (
                                isinstance(mapping_data, dict)
                                and "batch_mapping" in mapping_data
                            ):
                                batch_mapping = mapping_data["batch_mapping"]
                                files = mapping_data.get("files", [])
                                total_rows = mapping_data.get("total_rows", 0)
                                print(f"   - 包含 {len(batch_mapping)} 个批次映射")
                                print(f"   - 文件列表: {files}")
                                print(f"   - 总行数: {total_rows}")
                            else:
                                print(
                                    f"   - 旧格式映射，包含 {len(mapping_data)} 个批次映射"
                                )
                    else:
                        print(f"❌ 未找到批次映射文件: {mapping_path}")

                    # 检查文件信息
                    files_path = os.path.join(task_dir, "input_files.json")
                    if os.path.exists(files_path):
                        print(f"✅ 找到文件信息: {files_path}")
                        with open(files_path, "r", encoding="utf-8") as f:
                            files_data = json.load(f)
                            print(
                                f"   - 包含 {len(files_data.get('files', []))} 个文件信息"
                            )
                    else:
                        print(f"❌ 未找到文件信息: {files_path}")

                    # 检查每个文件的目录
                    for file_idx in range(len(input_files)):
                        file_dir = os.path.join(task_dir, f"file_{file_idx}")
                        if os.path.exists(file_dir):
                            print(f"✅ 找到文件{file_idx}目录: {file_dir}")
                            file_csv = os.path.join(file_dir, "original.csv")
                            if os.path.exists(file_csv):
                                print(f"   - 找到原始数据: {file_csv}")
                                try:
                                    df = pd.read_csv(file_csv)
                                    print(f"   - 包含 {len(df)} 行数据")
                                except Exception as e:
                                    print(f"   ❌ 读取原始数据失败: {e}")
                            else:
                                print(f"   ❌ 未找到原始数据: {file_csv}")
                        else:
                            print(f"❌ 未找到文件{file_idx}目录: {file_dir}")
                else:
                    print(f"❌ 未找到任务目录")
            else:
                print(f"❌ 预期外错误: {e}")
                print(traceback.format_exc())

        print("\n测试完成")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        print(traceback.format_exc())
    finally:
        # 清理测试文件
        print("\n=== 清理测试文件 ===")
        try:
            for file_path in input_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✓ 已删除: {file_path}")

            if os.path.exists("test_data/output.xlsx"):
                os.remove("test_data/output.xlsx")
                print(f"✓ 已删除: test_data/output.xlsx")

            # 可选：删除测试目录
            # shutil.rmtree("test_data")
            # print(f"✓ 已删除测试目录: test_data")
        except Exception as e:
            print(f"❌ 清理文件时出错: {e}")


if __name__ == "__main__":
    test_multi_file_process()
