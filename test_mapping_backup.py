#!/usr/bin/env python3

import os
import json
import pandas as pd
import uuid
import shutil
import time


def test_batch_mapping_backup():
    """测试批次映射备份与恢复机制"""
    print("\n=== 测试批次映射备份与恢复机制 ===")

    # 创建测试目录和数据
    task_id = f"test-mapping-{uuid.uuid4().hex[:8]}"
    test_dir = os.path.join("uploads", task_id)
    os.makedirs(test_dir, exist_ok=True)
    print(f"创建测试目录: {test_dir}")

    # 创建测试数据
    test_data = {
        "公司名称": ["测试公司1", "测试公司2", "测试公司3", "测试公司4", "测试公司5"],
        "联系人": ["张三", "李四", "王五", "赵六", "钱七"],
        "职位": ["经理", "总监", "工程师", "销售", "HR"],
        "电话": [
            "13800138000",
            "13900139000",
            "13700137000",
            "13600136000",
            "13500135000",
        ],
        "来源": ["test.xlsx"] * 5,
        "local_row_id": [str(uuid.uuid4()) for _ in range(5)],
    }
    df = pd.DataFrame(test_data)

    # 创建批次映射数据
    batch_size = 2
    total_rows = len(df)
    batch_mapping = {}

    # 模拟批次划分逻辑
    for batch_idx in range((total_rows + batch_size - 1) // batch_size):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_row_ids = df.iloc[start_idx:end_idx]["local_row_id"].tolist()
        batch_custom_id = f"batch_{batch_idx}_{len(batch_row_ids)}"
        batch_mapping[batch_custom_id] = batch_row_ids

    # 测试1：保存批次映射到JSON文件
    try:
        mapping_path = os.path.join(test_dir, "batch_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(batch_mapping, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存批次映射到: {mapping_path}")

        # 验证映射文件
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                loaded_mapping = json.load(f)

            if loaded_mapping == batch_mapping:
                print(f"✅ 映射文件保存和读取成功")
                for batch_id, row_ids in loaded_mapping.items():
                    print(f"  批次 {batch_id}: {len(row_ids)} 行")
            else:
                print(f"❌ 映射文件保存和读取失败: 数据不匹配")
    except Exception as e:
        print(f"❌ 测试批次映射保存失败: {e}")

    # 测试2：模拟批次输入文件丢失，但有映射备份的情况
    try:
        # 模拟结果数据
        batch_results = []
        for batch_id, row_ids in batch_mapping.items():
            # 创建模拟的百炼API返回结果
            results = []
            for i, row_id in enumerate(row_ids):
                result = {
                    "custom_id": batch_id,
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": f'[{{"公司名称":"测试公司{i+1}","联系人":"测试人员{i+1}","职位":"测试职位{i+1}","电话":"1380000{i+1}","行ID":"{row_id}"}}]'
                                    }
                                }
                            ]
                        }
                    },
                }
                batch_results.append(result)

        # 保存模拟结果文件
        result_path = os.path.join(test_dir, "batch_result.jsonl")
        with open(result_path, "w", encoding="utf-8") as f:
            for result in batch_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"✅ 已保存模拟结果文件: {result_path}")

        # 模拟处理逻辑
        print("\n模拟处理批次结果:")
        # 从映射文件恢复批次映射
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                recovered_mapping = json.load(f)
            print(f"✅ 成功从备份文件恢复批次映射")

            # 验证恢复的映射是否完整
            if len(recovered_mapping) == len(batch_mapping):
                print(f"✅ 恢复的映射完整，包含 {len(recovered_mapping)} 个批次")

                # 测试每个批次的行ID是否匹配
                match_count = 0
                for batch_id in batch_mapping:
                    if batch_id in recovered_mapping:
                        if set(batch_mapping[batch_id]) == set(
                            recovered_mapping[batch_id]
                        ):
                            match_count += 1

                print(f"✅ {match_count}/{len(batch_mapping)} 个批次的行ID完全匹配")
            else:
                print(
                    f"❌ 恢复的映射不完整: 原始 {len(batch_mapping)} 个批次, 恢复 {len(recovered_mapping)} 个批次"
                )
        except Exception as e:
            print(f"❌ 从备份文件恢复批次映射失败: {e}")

    except Exception as e:
        print(f"❌ 测试结果处理失败: {e}")

    # 测试3：模拟批次输入文件和映射备份都丢失的情况
    try:
        print("\n模拟原始数据重建映射:")
        # 删除现有的映射文件
        if os.path.exists(mapping_path):
            os.remove(mapping_path)
            print(f"已删除映射文件以模拟丢失情况")

        # 使用原始数据重建批次映射
        rebuilt_mapping = {}
        batch_size = 2  # 与原始批次大小保持一致

        for batch_idx in range((total_rows + batch_size - 1) // batch_size):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            batch_row_ids = batch_df["local_row_id"].tolist()
            batch_custom_id = f"batch_{batch_idx}_{len(batch_row_ids)}"
            rebuilt_mapping[batch_custom_id] = batch_row_ids

        print(f"✅ 成功从原始数据重建批次映射，包含 {len(rebuilt_mapping)} 个批次")

        # 验证重建的映射是否正确
        match_count = 0
        for batch_id in batch_mapping:
            if batch_id in rebuilt_mapping:
                if set(batch_mapping[batch_id]) == set(rebuilt_mapping[batch_id]):
                    match_count += 1

        print(f"✅ {match_count}/{len(batch_mapping)} 个批次的行ID完全匹配")

        # 保存重建的映射
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(rebuilt_mapping, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存重建的批次映射到: {mapping_path}")
    except Exception as e:
        print(f"❌ 测试重建映射失败: {e}")

    # 清理测试目录
    try:
        time.sleep(1)  # 确保文件不被占用
        shutil.rmtree(test_dir)
        print(f"✅ 已清理测试目录: {test_dir}")
    except Exception as e:
        print(f"❌ 清理测试目录失败: {e}")

    print("\n测试完成")


if __name__ == "__main__":
    test_batch_mapping_backup()
