#!/usr/bin/env python3

import os
import json
import uuid
import shutil
import time


def test_batch_mapping_simple():
    """测试批次映射文件的创建、读取和恢复机制 (不依赖pandas)"""
    print("\n=== 测试批次映射机制 (简化版) ===")

    # 创建测试目录
    task_id = f"test-simple-{uuid.uuid4().hex[:8]}"
    test_dir = os.path.join("uploads", task_id)
    os.makedirs(test_dir, exist_ok=True)
    print(f"创建测试目录: {test_dir}")

    # 创建模拟的行ID数据 (通常从DataFrame中获取)
    row_ids = [str(uuid.uuid4()) for _ in range(5)]
    print(f"创建测试行ID: {len(row_ids)} 行")

    # 创建批次映射数据
    batch_size = 2
    total_rows = len(row_ids)
    batch_mapping = {}

    # 模拟批次划分逻辑
    for batch_idx in range((total_rows + batch_size - 1) // batch_size):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_row_ids = row_ids[start_idx:end_idx]
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
                print(f"✅ 映射文件保存和读取成功，已验证映射完整性")
                for batch_id, batch_row_ids in loaded_mapping.items():
                    print(f"  批次 {batch_id}: {len(batch_row_ids)} 行")
            else:
                print(f"❌ 映射文件保存和读取失败: 数据不匹配")
    except Exception as e:
        print(f"❌ 测试批次映射保存失败: {e}")

    # 测试2：模拟创建批次输入JSONL文件
    try:
        jsonl_path = os.path.join(test_dir, "batch_input.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for batch_id, batch_row_ids in batch_mapping.items():
                # 创建批次记录
                record = {
                    "custom_id": batch_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "test-model"},
                    "row_ids": batch_row_ids,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ 已创建批次输入文件: {jsonl_path}")

        # 模拟从JSONL文件中提取批次映射
        extracted_mapping = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                custom_id = record.get("custom_id")
                record_row_ids = record.get("row_ids", [])
                if custom_id and record_row_ids:
                    extracted_mapping[custom_id] = record_row_ids

        # 验证提取的映射是否正确
        if extracted_mapping == batch_mapping:
            print(f"✅ 从JSONL文件提取批次映射成功，与原始映射完全匹配")
        else:
            print(f"❌ 从JSONL文件提取的批次映射与原始不匹配")
    except Exception as e:
        print(f"❌ 测试JSONL文件创建失败: {e}")

    # 测试3：模拟映射文件丢失后的恢复
    try:
        # 删除映射文件
        if os.path.exists(mapping_path):
            os.remove(mapping_path)
            print(f"模拟映射文件丢失")

        # 尝试从JSONL文件恢复
        if os.path.exists(jsonl_path):
            recovered_mapping = {}
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    custom_id = record.get("custom_id")
                    record_row_ids = record.get("row_ids", [])
                    if custom_id and record_row_ids:
                        recovered_mapping[custom_id] = record_row_ids

            # 保存恢复的映射
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(recovered_mapping, f, ensure_ascii=False, indent=2)
            print(f"✅ 已从JSONL文件恢复并保存批次映射")

            # 验证恢复的映射是否正确
            if recovered_mapping == batch_mapping:
                print(f"✅ 恢复的批次映射与原始映射完全匹配")
            else:
                print(f"❌ 恢复的批次映射与原始不匹配")
        else:
            print(f"❌ JSONL文件不存在，无法恢复")
    except Exception as e:
        print(f"❌ 测试映射恢复失败: {e}")

    # 清理测试目录
    try:
        time.sleep(1)  # 确保文件不被占用
        shutil.rmtree(test_dir)
        print(f"✅ 已清理测试目录: {test_dir}")
    except Exception as e:
        print(f"❌ 清理测试目录失败: {e}")

    print("\n测试完成")


if __name__ == "__main__":
    test_batch_mapping_simple()
