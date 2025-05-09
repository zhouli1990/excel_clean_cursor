#!/usr/bin/env python3

import os
import pandas as pd
import uuid
import traceback
from processor import create_bailian_batch_input, submit_bailian_batch_job


def test_batch_input_creation():
    """测试批量输入文件创建和提交功能"""
    try:
        # 创建一个简单的测试DataFrame
        test_data = {
            "公司名称": ["测试公司1", "测试公司2", "测试公司3"],
            "联系人": ["张三", "李四", "王五"],
            "职位": ["经理", "总监", "工程师"],
            "电话": ["13800138000", "13900139000", "13700137000"],
            "来源": ["测试.xlsx", "测试.xlsx", "测试.xlsx"],
            "local_row_id": [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())],
        }
        df = pd.DataFrame(test_data)

        # 创建测试任务ID
        task_id = f"test-{uuid.uuid4().hex[:8]}"

        # 测试配置
        config = {
            "TARGET_COLUMNS": ["公司名称", "联系人", "职位", "电话", "来源"],
            "DASHSCOPE_API_KEY": "sk-test-key",  # 使用测试key，不会真正提交请求
            "BAILIAN_MODEL_NAME": "qwen-plus",
            "BAILIAN_COMPLETION_WINDOW": "24h",
            "BATCH_SIZE": 2,
        }

        print("\n=== 测试批次输入文件创建 ===")
        # 调用函数创建JSONL输入文件
        jsonl_path = create_bailian_batch_input(df, task_id, config)

        # 验证文件是否已创建
        if os.path.exists(jsonl_path):
            print(f"✅ 成功创建批次输入文件: {jsonl_path}")
            # 检查文件内容
            with open(jsonl_path, "r", encoding="utf-8") as f:
                content = f.readlines()
                print(f"  文件包含 {len(content)} 行")

            # 检查task_dir中是否创建了必要文件
            task_dir = os.path.join("uploads", task_id)
            print(f"  任务目录: {task_dir}")
            if os.path.exists(task_dir):
                files = os.listdir(task_dir)
                print(f"  目录中的文件: {files}")

            # 检查是否按batch_size分了批次
            expected_batches = (len(df) + config["BATCH_SIZE"] - 1) // config[
                "BATCH_SIZE"
            ]
            print(f"  预期批次数: {expected_batches}, 实际行数: {len(content)}")

            return jsonl_path, task_id, config
        else:
            print(f"❌ 批次输入文件未创建: {jsonl_path}")
            return None, task_id, config

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        print(traceback.format_exc())
        return None, None, None


def main():
    jsonl_path, task_id, config = test_batch_input_creation()
    if jsonl_path:
        print("\n=== 测试任务提交过程中的目录操作 ===")
        try:
            # 我们不会真正提交任务，只测试本地文件操作部分
            # 设置一个明显错误的API key使其在API请求前就失败
            config["DASHSCOPE_API_KEY"] = "sk-test-api-key"

            try:
                batch_id = submit_bailian_batch_job(jsonl_path, task_id, config)
                print(f"✅ 测试通过: batch_id = {batch_id}")
            except Exception as e:
                # 我们期望API调用会失败，但本地文件操作应该完成
                print(f"预期的API错误: {e}")

                # 检查是否正确复制了batch_input.jsonl
                task_dir = os.path.join("uploads", task_id)
                local_copy = os.path.join(task_dir, "batch_input.jsonl")
                if os.path.exists(local_copy):
                    print(f"✅ 成功保存JSONL本地副本: {local_copy}")
                else:
                    print(f"❌ 未找到JSONL本地副本: {local_copy}")
        except Exception as e:
            print(f"❌ 测试提交过程中发生错误: {e}")
            print(traceback.format_exc())

    print("\n测试完成")


if __name__ == "__main__":
    main()
