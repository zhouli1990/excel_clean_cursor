#!/usr/bin/env python3

import os
import pandas as pd
import uuid
import json
import traceback
import time
import shutil
from processor import (
    create_bailian_batch_input,
    submit_bailian_batch_job,
    download_and_process_bailian_results,
)


def test_fixed_functions():
    """测试修复后的功能"""
    print("\n=== 测试修复后的百炼API批处理功能 ===")

    # 创建测试数据
    try:
        # 创建测试DataFrame
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
        task_id = f"test-fix-{uuid.uuid4().hex[:8]}"

        # 测试配置
        config = {
            "TARGET_COLUMNS": ["公司名称", "联系人", "职位", "电话", "来源"],
            "DASHSCOPE_API_KEY": "sk-test-key",  # 使用测试key，不会真正提交请求
            "BAILIAN_MODEL_NAME": "qwen-plus",
            "BAILIAN_COMPLETION_WINDOW": "24h",
            "BATCH_SIZE": 2,
        }

        print(f"\n1. 测试修复的submit_bailian_batch_job函数")
        # 创建测试目录
        test_dir = os.path.join("uploads", task_id)
        os.makedirs(test_dir, exist_ok=True)

        # 创建测试JSONL文件
        test_jsonl = os.path.join(test_dir, "batch_input.jsonl")
        with open(test_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"test": "data"}, ensure_ascii=False) + "\n")

        print(f"  创建测试JSONL文件: {test_jsonl}")

        # 测试调用submit_bailian_batch_job (预期会在API调用前失败)
        try:
            submit_bailian_batch_job(test_jsonl, task_id, config)
        except Exception as e:
            # 预期会失败，但应该是在API调用阶段，而不是文件操作阶段
            if "百炼API Key无效" in str(e):
                print(f"  ✅ 测试通过: 预期的API错误: {e}")
            else:
                print(f"  ❌ 测试失败: 预期API Key错误，但收到: {e}")

        print(f"\n2. 测试JSON解析和行ID映射")
        # 创建模拟的批次结果
        test_result_data = [
            {
                "custom_id": "batch_0_2",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": f'[{{"公司名称":"测试公司1","联系人":"张三","职位":"经理","电话":"13800138000","行ID":"{df.iloc[0]["local_row_id"]}"}},{{"公司名称":"测试公司2","联系人":"李四","职位":"总监","电话":"13900139000","行ID":"{df.iloc[1]["local_row_id"]}"}}]'
                                }
                            }
                        ]
                    }
                },
            }
        ]

        # 创建批次结果文件
        test_result_file = os.path.join(test_dir, "batch_result.jsonl")
        with open(test_result_file, "w", encoding="utf-8") as f:
            for record in test_result_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  创建测试结果文件: {test_result_file}")

        # 创建批次映射文件
        with open(test_jsonl, "w", encoding="utf-8") as f:
            batch_record = {
                "custom_id": "batch_0_2",
                "row_ids": [df.iloc[0]["local_row_id"], df.iloc[1]["local_row_id"]],
            }
            f.write(json.dumps(batch_record, ensure_ascii=False) + "\n")

        print(f"  更新测试批次映射: {test_jsonl}")

        # 测试处理结果
        try:
            result_df = download_and_process_bailian_results(
                "test-batch-id", df.iloc[:2], task_id, config
            )
            if not result_df.empty:
                print(f"  ✅ 测试通过: 成功处理结果，得到 {len(result_df)} 行数据")
                print(f"  结果列: {list(result_df.columns)}")
            else:
                print(f"  ❌ 测试失败: 处理结果为空DataFrame")
        except Exception as e:
            print(f"  ❌ 测试失败: 处理结果时出错: {e}")
            print(traceback.format_exc())

        # 清理测试文件和目录
        time.sleep(1)  # 确保文件不被占用
        try:
            shutil.rmtree(test_dir)
            print(f"  已清理测试目录: {test_dir}")
        except Exception as e:
            print(f"  无法清理测试目录: {e}")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        print(traceback.format_exc())

    print("\n测试完成")


if __name__ == "__main__":
    test_fixed_functions()
