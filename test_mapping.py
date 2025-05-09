#!/usr/bin/env python3

import os
import json
import traceback


def test_row_id_mapping():
    """测试行ID映射和结果合并功能"""
    print("\n=== 测试行ID映射和结果合并 ===")

    # 测试目录
    task_dir = "uploads/67c96a39-a281-4708-9049-4ec862d141d6"

    # 手动创建输入映射文件进行测试
    input_jsonl = os.path.join(task_dir, "test_batch_input.jsonl")

    # 读取结果文件的第一行来获取custom_id和内容
    result_file = os.path.join(task_dir, "batch_result.jsonl")
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return

    print("从结果文件获取第一个batch的数据")
    with open(result_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        record = json.loads(first_line)
        custom_id = record.get("custom_id", "batch_0_0")

        content = None
        if "response" in record and "body" in record["response"]:
            content = (
                record["response"]["body"]["choices"][0]
                .get("message", {})
                .get("content", "")
            )
        elif "body" in record and "choices" in record["body"]:
            content = record["body"]["choices"][0].get("message", {}).get("content", "")

        if not content:
            print("❌ 结果中没有找到content字段")
            return

        # 解析content
        content = content.strip()
        try:
            # 尝试解析JSON数组
            llm_results = json.loads(content)
            if not isinstance(llm_results, list):
                print(f"❌ 解析结果不是数组: {type(llm_results)}")
                return

            # 创建测试行ID列表
            test_row_ids = [
                item.get("行ID", f"row-{i}") for i, item in enumerate(llm_results[:5])
            ]
            print(f"使用前5行的行ID进行测试: {test_row_ids}")

            # 创建测试批次映射文件
            with open(input_jsonl, "w", encoding="utf-8") as f:
                # 写入一个测试批次记录
                test_batch = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "qwen-plus"},
                    "row_ids": test_row_ids,
                }
                f.write(json.dumps(test_batch, ensure_ascii=False))

            print(f"✅ 已创建测试批次映射文件: {input_jsonl}")

            # 测试批次映射加载
            print("\n模拟批次映射加载过程:")
            batch_row_ids_map = {}

            with open(input_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        batch_custom_id = record.get("custom_id")
                        row_ids = record.get("row_ids", [])
                        if batch_custom_id and row_ids:
                            batch_row_ids_map[batch_custom_id] = row_ids
                    except Exception as e:
                        print(f"❌ 读取输入文件时出错: {e}")
                        continue

            # 打印找到的批次映射关系
            print(f"✅ 找到 {len(batch_row_ids_map)} 个批次的行ID映射")
            for batch_id, row_ids in batch_row_ids_map.items():
                print(f"批次 {batch_id} 包含 {len(row_ids)} 个行ID")

            # 模拟结果与行ID的匹配过程
            print("\n测试结果与行ID的匹配过程:")
            matched_results = []
            for i, result_item in enumerate(llm_results[:5]):
                # 直接使用索引关联行ID(简化模拟)
                if i < len(test_row_ids):
                    row_id = test_row_ids[i]
                    print(f"结果项 {i+1} 匹配行ID: {row_id}")
                    result_item["local_row_id"] = row_id
                    matched_results.append(result_item)
                else:
                    print(f"⚠️ 结果项 {i+1} 没有对应的行ID")

            print(f"✅ 成功匹配 {len(matched_results)}/{len(llm_results[:5])} 个结果项")

            # 清理测试文件
            os.remove(input_jsonl)
            print(f"✅ 已清理测试文件: {input_jsonl}")

        except Exception as e:
            print(f"❌ JSON解析失败: {e}")
            return

    print("\n测试完成")


if __name__ == "__main__":
    test_row_id_mapping()
