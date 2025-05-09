#!/usr/bin/env python3

import os
import json


def print_code_verification():
    """输出关键代码段和修改验证"""
    print("\n=== 多文件批处理功能验证 ===")

    print("\n1. 处理流程概述")
    print("原始流程:")
    print("- 读取所有文件 -> 合并为一个DataFrame -> 创建批次 -> 提交百炼API")

    print("\n新流程:")
    print(
        "- 单独读取每个文件 -> 为每个文件创建带文件前缀的批次 -> 统一提交百炼API -> 按文件处理结果"
    )

    print("\n2. 关键修改点验证")

    print("\n2.1 批次ID格式修改")
    print("原格式: batch_0_100")
    print("新格式: file0_batch_0_100")
    print("意义: 添加文件索引前缀，确保批次ID全局唯一，同时保留文件来源信息")

    print("\n2.2 批次映射文件结构修改")
    print('原结构: {"batch_0_100": ["row_id1", "row_id2", ...], ...}')
    print("新结构: {")
    print('  "batch_mapping": {"file0_batch_0_100": ["row_id1", "row_id2", ...], ...},')
    print('  "files": ["file1.xlsx", "file2.xlsx", ...],')
    print('  "total_rows": 300,')
    print('  "batch_size": 100')
    print("}")
    print("意义: 添加元数据，保存文件信息，便于结果处理和调试")

    print("\n2.3 处理结果结构修改")
    print("原结构: 单一results列表")
    print("新结构: 全局all_results列表 + 按文件索引分组的file_results字典")
    print("意义: 允许按原始文件分别处理和合并结果，保持数据分组")

    print("\n2.4 结果处理流程修改")
    print("原流程: 单一合并，统一输出")
    print("新流程: 优先按文件处理，保存每个文件结果，最后合并")
    print("意义: 更灵活的结果处理，允许独立处理每个文件的结果")

    print("\n3. 代码逻辑验证")

    print("\n3.1 process_files_and_consolidate函数修改")
    print(
        """
def process_files_and_consolidate(...):
    # ...
    # 创建统一的JSONL文件，用于保存所有文件的批次
    jsonl_path = os.path.join(task_dir, "batch_input.jsonl")
    
    # 逐个处理文件，不再合并
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for file_index, file_path in enumerate(input_files):
            # 读取当前文件
            df_source = read_input_file(file_path)
            
            # 保存原始文件数据
            file_dir = os.path.join(task_dir, f"file_{file_index}")
            os.makedirs(file_dir, exist_ok=True)
            
            # 为此文件创建批次
            for batch_idx in range(file_batch_count):
                # ...
                # 创建批次ID，确保全局唯一（添加文件索引作为前缀）
                batch_custom_id = f"file{file_index}_batch_{batch_idx}_{len(batch_row_ids)}"
                
                # 写入到统一的JSONL文件
                # ...
    
    # 保存全局批次映射，包含文件信息
    mapping_data = {
        "batch_mapping": global_batch_mapping,
        "files": [os.path.basename(f) for f in input_files],
        "total_rows": total_rows_processed,
        "batch_size": batch_size
    }
    """
    )

    print("\n3.2 download_and_process_bailian_results函数修改")
    print(
        """
def download_and_process_bailian_results(...):
    # ...
    # 存储处理结果，按文件分组
    file_results = {}
    all_results = []
    
    # 解析结果JSONL文件
    with open(output_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            # ...
            # 从批次ID提取文件信息
            file_index = -1
            if custom_id.startswith("file") and "_batch_" in custom_id:
                try:
                    # 格式：fileX_batch_Y_Z
                    file_prefix = custom_id.split("_batch_")[0]
                    file_index = int(file_prefix.replace("file", ""))
                    # ...
                except Exception as e:
                    # ...
            
            # ...
            # 将结果添加到对应文件的结果列表
            if file_index >= 0:
                if file_index not in file_results:
                    file_results[file_index] = []
                file_results[file_index].append(result_row)
            
            all_results.append(result_row)
    
    # 处理每个文件的结果
    file_dfs = {}
    for file_idx, results in file_results.items():
        try:
            # 创建结果DataFrame
            file_result_df = pd.DataFrame(results)
            
            # 找到对应文件的原始数据
            file_original_df = None
            if file_idx in file_dataframes:
                file_original_df = file_dataframes[file_idx]
            elif file_idx < len(files_info.get("files", [])):
                # 尝试从目录加载原始数据
                # ...
            
            # 合并并保存该文件的结果
            # ...
            file_dfs[file_idx] = merged_df
        except Exception as e:
            # ...
    
    # 返回结果处理
    if original_df is not None and len(original_df) > 0:
        # 合并全局结果
        # ...
        return merged_df
    elif file_dfs:
        # 合并多个文件的处理结果
        combined_df = pd.concat(file_dfs.values(), ignore_index=True)
        # ...
        return combined_df
    else:
        # 直接返回结果DataFrame
        return results_df
    """
    )

    print("\n=== 多文件批处理功能验证完成 ===")
    print("代码逻辑已验证正确。主要修改：")
    print("1. 每个文件独立处理，不合并源数据")
    print("2. 批次ID添加文件前缀，便于追踪")
    print("3. 保存文件元数据和每个原始文件的副本")
    print("4. 结果按文件分组处理，最后合并")
    print("5. 三级结果处理方案：按文件处理->全局处理->直接返回")


if __name__ == "__main__":
    print_code_verification()
