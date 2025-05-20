# -*- coding: utf-8 -*-
import requests
import pandas as pd
import time
import numpy as np
import os
import shutil
import json
import traceback  # 用于打印详细的错误堆栈信息
import math  # 用于计算批处理数量
import uuid  # Added import
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import openai  # 新增: 阿里云百炼 OpenAI 兼容接口
import datetime  # Added import
from utils.logger import setup_logger

# === Logger Setup ===
logger = setup_logger("processor")

# === 配置项 (这些值现在由 app.py 传入) ===
# DEEPSEEK_API_KEY = "YOUR_API_KEY"       # 从 app.py 获取
# TARGET_COLUMNS = [...]                  # 从 app.py 获取
# BATCH_SIZE = 160                        # 从 app.py 获取
# MAX_COMPLETION_TOKENS = 8192            # 从 app.py 获取
# DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/chat/completions" # 从 app.py 获取
# DASHSCOPE_API_KEY = ""                  # 从 app.py 获取
# BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 从 app.py 获取
# BAILIAN_BATCH_ENDPOINT = "/v1/chat/completions" # 从 app.py 获取
# BAILIAN_MODEL_NAME = "qwen-turbo-latest"        # 从 app.py 获取
# BAILIAN_COMPLETION_WINDOW = "24h"       # 从 app.py 获取


def read_input_file(
    file_path: str,  # required_columns: Optional[List[str]] = None # Removed this parameter
) -> pd.DataFrame:
    """
    Reads an Excel or CSV file into a Pandas DataFrame.
    # Removed validation of required columns.
    Adds a unique 行ID to each row if needed.
    *** Removed adding '来源' column here. ***
    """
    logger.info(f"Reading file: {file_path}")
    try:
        if file_path.endswith((".xlsx", ".xls")):
            # 添加converters参数确保电话列为字符串
            # 注意：由于这里无法访问config，使用所有可能的电话列名作为字符串转换
            possible_phone_cols = [
                "电话",
                "手机",
                "电话号码",
                "手机号码",
                "联系方式",
                "电话/手机",
            ]
            converters = {col: str for col in possible_phone_cols}
            all_sheets = pd.read_excel(
                file_path, sheet_name=None, engine="openpyxl", converters=converters
            )
            df = pd.concat(all_sheets.values(), ignore_index=True).astype(str)
        elif file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {file_path}, trying gbk.")
                df = pd.read_csv(file_path, encoding="gbk")
        else:
            raise ValueError(
                "Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv)."
            )

        logger.info(f"Successfully read {len(df)} rows from {file_path}.")

        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        logger.debug(f"Original columns: {list(df.columns)}")

        # Validate required columns - REMOVED
        # if required_columns:
        #     missing_cols = [col for col in required_columns if col not in df.columns]
        #     if missing_cols:
        #         raise ValueError(f"Missing required columns in {file_path}: {', '.join(missing_cols)}")
        #     logger.info(f"All required columns {required_columns} found in {file_path}.")

        # Add 行ID instead of local_row_id
        if "行ID" not in df.columns:
            df["行ID"] = [str(uuid.uuid4()) for _ in range(len(df))]
            logger.info(f"Added '行ID' column to DataFrame from {file_path}.")
        else:
            # Handle case where column might exist but contain NaNs or duplicates
            existing_ids = df["行ID"].dropna().astype(str)
            if len(existing_ids) != len(df) or existing_ids.duplicated().any():
                logger.warning(
                    f"'行ID' column exists in {file_path} but contains nulls or duplicates. Regenerating IDs."
                )
                df["行ID"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Add source information ('来源' column)
        df["来源"] = os.path.basename(file_path)
        logger.info(
            f"Added source filename as '来源' column: {os.path.basename(file_path)}"
        )

        # Fill NaN values with empty strings BEFORE processing
        df = df.fillna("")  # Important: Prevents issues with NaN in string operations

        return df

    except FileNotFoundError:
        # 文件路径不存在
        print(f"   ❌ 文件未找到: {file_path}")
        return None
    except ImportError as e:
        # 缺少核心库 (通常是 pandas)
        print(f"   ❌ 缺少必要的库: {e}。请确保已安装 pandas, openpyxl, xlrd。")
        return None
    except Exception as e:
        # 捕获所有其他未预料到的异常
        print(f"   ❌ 读取文件 '{file_path}' 时发生未知错误:")
        # 打印详细的错误堆栈信息
        print(traceback.format_exc())
        return None


def create_bailian_batch_input(df: pd.DataFrame, task_id: str, config: dict) -> str:
    """
    为阿里云百炼Batch API创建输入文件。
    将DataFrame中的数据按BATCH_SIZE分批处理，每批创建一个请求，并写入JSONL文件。

    Args:
        df: 包含需要处理的数据的DataFrame。
        task_id: 任务ID，用于创建输出目录。
        config: 包含配置信息的字典。

    Returns:
        str: 创建的JSONL文件路径。
    """
    print(f"   >> 开始准备百炼Batch API输入文件...")

    # 获取目标列和模型名称配置
    target_columns_config = config.get(
        "TARGET_COLUMNS", ["公司名称", "联系人", "职位", "电话", "来源"]
    )
    source_column_name = "来源"
    model_name = config.get("BAILIAN_MODEL_NAME", "qwen-turbo-latest")
    batch_endpoint = config.get("BAILIAN_BATCH_ENDPOINT", "/v1/chat/completions")
    max_tokens = config.get("MAX_COMPLETION_TOKENS", 8192)
    # 获取批处理大小
    batch_size = config.get("BATCH_SIZE", 100)

    # 准备给LLM的目标列(不含来源)
    llm_target_columns = [
        col for col in target_columns_config if col != source_column_name
    ]

    # 确保任务目录存在
    task_dir = os.path.join("uploads", task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 创建JSONL文件路径
    jsonl_path = os.path.join(task_dir, "batch_input.jsonl")

    # 获取源列标题
    source_headers = df.columns.astype(str).tolist()
    source_headers_json = json.dumps(source_headers, ensure_ascii=False)

    # 构建系统提示词(system prompt)
    target_schema_json = json.dumps(llm_target_columns, ensure_ascii=False)
    system_content = f"""
你是一个专业的数据处理引擎。你的核心任务是从用户提供的源数据批次 (Source Batch Data) 中，为每一行数据自动推断并提取与目标模式 (Target Schema) 字段最匹配的信息，并严格按照指定的目标模式进行标准化。
**重要要求：**
1. 你的输出必须是一个有效的 JSON 数组 (列表)，数组中的每个元素都是一个对应输入行的、符合 Target Schema 的 JSON 对象。元素的数量必须与输入数组完全一致，除非去重规则生效。
2. 每个结果对象必须包含"来源"字段，且其值取自输入的来源表格名称。
3. 返回的所有列标题（字段名）必须与 Target Schema 完全一致、顺序一致，不允许多余或缺失字段。
4. 找不到的字段请置空字符串。

**目标模式 (Target Schema):**
```json
{target_schema_json}
```
**核心处理规则：**
**1. 电话号码规范化与格式转换：**
    - 原始电话号码中的所有非数字字符（例如：连字符 "-", 空格, 括号 "()" 等）都必须被去除，只保留纯数字序列。
    - 例如，如果输入为 "133-0437-0109"，则输出应为 "13304370109"。

**2. 企业名称清洗规则：**
   - 只保留企业名称的核心主体部分。
   - 必须去除所有附加描述性信息，例如：括号内的注释（如"（广东省500强）"）、注册资本信息（如"注册资本 1000万美元"）、荣誉称号、规模描述、地理位置描述等。
   - **正例**：
     - 企业名称输入："广州立邦涂料有限公司，（广东省500强）注册资本 1000万美元"
     - 企业名称输出："广州立邦涂料有限公司"
   - **反例**：
     - 错误输出："广州立邦涂料有限公司，（广东省500强）注册资本 1000万美元"

**3. 多行拆分与混杂内容处理规则：**
   - **目标**：如果单个输入行包含多个独立的联系人信息单元，则必须将该行拆分为多个输出行，每个输出行对应一个联系人信息单元。
   - **识别混杂内容**：仔细检查所有输入字段（尤其是"职务"、"备注"等自由文本字段）以识别独立的联系人实体。一个联系人实体通常由姓名、电话号码和/或职位信息组成。
   - **拆分逻辑**：
     - 为每个从混杂内容中识别出的独立联系人实体生成一条新的输出行。
     - 如果"电话"或"手机号"列明确包含由分隔符（支持的分隔符包括：中英文分号（;；）、中英文逗号（,，）、空格、斜杠（/））分隔的多个号码，并且没有更具体的联系人信息与之对应，也应为每个号码生成一条新的输出行。
     - 当"混杂内容处理规则"识别出多个完整联系人实体时，其拆分逻辑优先。
   - **信息填充**：对于每个生成的输出行，应填充从混杂内容中提取出的对应实体的姓名、电话和职位到 Target Schema 的相应字段。
   - **信息复制**：其他原始行级别的信息（如"公司名称"、"来源"、"行ID"）必须复制到所有由此输入行生成的新输出行中。
   - **输入示例** (处理混杂内容)：
     ```json
    {{ // 注意: f-string 中的 JSON 示例，花括号需要转义
    "序号": 83691,
    "公司名称": "广州仕邦人力资源有限公司",
    "职务": "18520784668  IT 谢生（昀昌）    020-66393586 财务部     财务总监为张生，",
    "姓名": "",
    "电话": "",
    "行ID": "c3b4e24d-4d31-4102-aac2-0fbbb8825e7d",
    "来源": "特殊场景测试2（电话联系人在一列）.xlsx"
    }}
     ```
   - **输出示例** (对应上述输入，生成两条记录)：
     ```json
     [
       {{ // 注意: f-string 中的 JSON 示例，花括号需要转义
        "公司名称": "广州仕邦人力资源有限公司",
        "姓名": "谢生（昀昌）",
        "电话": "18520784668", // 规范化后
        "职位": "IT",
        "行ID": "c3b4e24d-4d31-4102-aac2-0fbbb8825e7d",
        "来源": "特殊场景测试2（电话联系人在一列）.xlsx"
        }},
       {{ // 注意: f-string 中的 JSON 示例，花括号需要转义
        "公司名称": "广州仕邦人力资源有限公司",
        "姓名": "张生",
        "电话": "02066393586", // 规范化后
        "职位": "财务总监",
        "行ID": "c3b4e24d-4d31-4102-aac2-0fbbb8825e7d",
        "来源": "特殊场景测试2（电话联系人在一列）.xlsx"
        }}
     ]
     ```
**4. 基于"企业名称"和"电话"的输出去重规则：**
   - **目标**：确保在你最终生成的JSON数组中，对于拥有相同"企业名称"和规范化后"电话号码"的组合，只包含一条唯一的记录。
   - **处理逻辑**：当你处理输入数据并生成结果对象时，如果一个结果对象其"企业名称"和规范化后的"电话号码"与你在此次任务中*已经生成并计划包含在最终输出数组中*的某个对象完全相同，则应避免再次添加这个重复的对象。对于一组重复记录，请只保留你遇到的**第一条有效记录**。
   - **关键字段**：去重基于"企业名称"和"电话"这两个字段的组合。电话号码在比较前必须是经过规范化（仅含数字，已通过规则1处理）的。
   - **示例**：
     - 假设原始输入经过初步处理（包括拆分、规范化）后，可能产生如下中间结果，准备合并到最终输出：
       ```
       [
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "郭间荷", "职位": "国家税务总局佛山市南海区税务局", "电话": "15015608915", "行ID": "84167", "来源": "file.xlsx"}},
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "张妍", "职位": "", "电话": "86222923", "行ID": "84167", "来源": "file.xlsx"}}, // 不同电话，保留
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "刘永雄", "职位": "", "电话": "13802628974", "行ID": "84168", "来源": "file.xlsx"}},
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "郭间荷", "职位": "国家税务总局佛山市南海区税务局", "电话": "15015608915", "行ID": "84169", "来源": "file.xlsx"}}, // 与第一条的企业名称和电话重复
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "刘永雄", "职位": "", "电话": "15015608915", "行ID": "84170", "来源": "file.xlsx"}}  // 与第一条的企业名称和电话重复，但联系人不同
       ]
       ```
     - **期望的去重后输出（单个JSON数组）**：
       ```json
       [
         {{
           "企业名称": "广东世纪达建设集团有限公司",
           "联系人": "郭间荷",
           "职位": "国家税务总局佛山市南海区税务局",
           "电话": "15015608915", // 规范化后
           "行ID": "84167",
           "来源": "file.xlsx"
         }},
         {{
           "企业名称": "广东世纪达建设集团有限公司",
           "联系人": "张妍",
           "职位": "",
           "电话": "86222923", // 规范化后
           "行ID": "84167", // 假设行ID允许在拆分或关联时复用
           "来源": "file.xlsx"
         }},
         {{
           "企业名称": "广东世纪达建设集团有限公司",
           "联系人": "刘永雄",
           "职位": "",
           "电话": "13802628974", // 规范化后
           "行ID": "84168",
           "来源": "file.xlsx"
         }}
         // 后续两条与第一条"企业名称"和"电话"重复的记录被去除，即使联系人字段可能不同，因为我们优先保留第一条。
       ]
       ```
   - **上下文**：此去重逻辑应用于你处理当前整个任务（即一个完整的源数据批次 Source Batch Data）所产生的所有记录，目标是确保最终返回的JSON数组内不含基于（企业名称，电话）的重复项。

**通用处理指南：**
1.  **自动推断与映射**：结合源数据内容，自动推断并提取与 Target Schema 字段最匹配的信息。
2.  **字段严格匹配**：只输出 Target Schema 中定义的字段，输出顺序与 Target Schema 保持一致，不要输出任何多余字段。
3.  **格式整理**：去除字段值中不必要的首尾空格。对于文本内容，避免去除关键的内部标点，除非特定规则要求（如电话号码规范化）。
4.  **批量处理意识**：你将收到一个包含多行数据的数组，你需要独立处理每一行（可能将其拆分为多行或根据去重规则舍弃），并将所有最终生成的记录合并到一个JSON数组中返回。
5.  **最终输出格式**：只返回一个顶级的JSON数组，数组内包含所有处理后的记录对象。绝对不要在JSON数组之外添加任何额外的解释、注释、markdown标记或任何非JSON文本。
"""

    # 按照batch_size拆分数据
    total_rows = len(df)
    batch_count = (total_rows + batch_size - 1) // batch_size  # 向上取整
    print(
        f"   >> 将 {total_rows} 行数据分为 {batch_count} 个批次，每批最多 {batch_size} 行"
    )

    # 创建批次ID到行ID的映射，用于后续恢复
    batch_mapping = {}

    # 将DataFrame的批次转换为请求，写入JSONL文件
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            # 获取批次中所有行的行ID，用作custom_id
            batch_row_ids = batch_df["行ID"].tolist()
            batch_custom_id = f"batch_{batch_idx}_{len(batch_row_ids)}"

            # 保存批次ID到行ID的映射
            batch_mapping[batch_custom_id] = batch_row_ids

            # 将批次数据转换为用户可理解的文本格式
            batch_content = []
            for _, row in batch_df.iterrows():
                row_str = "\n".join(
                    [f"{k}: {v}" for k, v in row.items() if k != "行ID"]
                )
                # 添加行ID作为行标识
                row_str += f"\n行ID: {row['行ID']}"
                batch_content.append(row_str)

            # 合并所有行内容
            all_rows_content = "\n\n--- 行分隔符 ---\n\n".join(batch_content)

            # 获取源文件名
            if "source_file" in batch_df.columns:
                source_filename = batch_df["source_file"].iloc[0]
            else:
                # 尝试从batch_row_ids获取原始文件名
                source_filename = local_id_to_source_map.get(
                    batch_row_ids[0], "未知来源"
                )

            # 记录使用的源文件名
            print(f"   >>> 批次 {batch_idx+1} 使用的源文件名: {source_filename}")

            # 用户提示词(user prompt)
            user_content = f"""
这是本次处理的数据来源: {source_filename}

这是本次需要处理的源数据表格的列标题:
{source_headers_json}

这是需要处理的 {len(batch_df)} 行源数据:
{all_rows_content}

请根据你在System指令中被赋予的角色和规则处理这些数据，并返回标准化的JSON数组。
每个处理结果必须包含原始行的"行ID"值，以便我们能够追踪结果与原始数据的对应关系。
同时，在每个结果中添加"来源"字段，值必须完全设为"{source_filename}"，不要对文件名做任何修改。
"""

            # 构建消息数组
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            # 构建API请求体
            body = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 1.0,  # 可配置
            }

            # 构建JSONL记录
            record = {
                "custom_id": batch_custom_id,  # 使用批次ID作为custom_id
                "method": "POST",
                "url": batch_endpoint,
                "body": body,
                "row_ids": batch_row_ids,  # 保存批次中所有行的row_id
            }

            # 写入JSONL文件
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 保存批次映射到单独的文件，作为备份
    mapping_path = os.path.join(task_dir, "batch_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(batch_mapping, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 已保存批次映射备份到: {mapping_path}")

    print(
        f"   ✅ 已成功创建百炼Batch API输入文件: {jsonl_path} (包含 {batch_count} 个批次请求)"
    )
    return jsonl_path


def submit_bailian_batch_job(jsonl_path: str, task_id: str, config: dict) -> str:
    """
    提交阿里云百炼Batch任务。

    Args:
        jsonl_path: JSONL输入文件路径。
        task_id: 任务ID，使用时间戳格式(如task_20250506_174440)。
        config: 包含配置信息的字典。

    Returns:
        str: Batch任务ID。
    """
    print(f"   >> 开始提交百炼Batch任务...")

    # 从嵌套结构中获取llm_config
    llm_config = config.get("llm_config", {})

    # 获取配置
    api_key = llm_config.get("DASHSCOPE_API_KEY", "")
    base_url = llm_config.get(
        "BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    batch_endpoint = llm_config.get("BAILIAN_BATCH_ENDPOINT", "/v1/chat/completions")
    completion_window = llm_config.get("BAILIAN_COMPLETION_WINDOW", "24h")

    if not api_key or not api_key.startswith("sk-"):
        print("   ❌ 错误: 百炼API Key未提供或格式无效。")
        raise ValueError("百炼API Key无效")

    try:
        # 确保任务目录存在
        task_dir = os.path.join("uploads", task_id)
        os.makedirs(task_dir, exist_ok=True)

        # 确保全局映射目录存在
        global_mapping_dir = os.path.join("uploads", "global_mappings")
        os.makedirs(global_mapping_dir, exist_ok=True)

        # 保存JSONL文件本地副本用于后续处理
        local_copy_path = os.path.join(task_dir, "batch_input.jsonl")
        # 检查源文件路径和目标路径是否相同
        if jsonl_path != local_copy_path:
            try:
                shutil.copy(jsonl_path, local_copy_path)
                print(f"   ✓ 已保存JSONL文件本地副本: {local_copy_path}")
            except Exception as e:
                print(f"   ⚠️ 复制JSONL文件失败: {e}，将直接使用原始文件")
                local_copy_path = jsonl_path
        else:
            print(f"   ✓ JSONL文件已在任务目录中: {local_copy_path}")

        # 创建和存储批次映射
        batch_mapping = {}
        row_to_source_mapping = {}

        # 从输入文件提取批次映射和行ID到源文件映射
        try:
            with open(local_copy_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        custom_id = record.get("custom_id")
                        row_ids = record.get("row_ids", [])
                        source_file = record.get("source_file", "未知来源")

                        if custom_id and row_ids:
                            batch_mapping[custom_id] = row_ids

                        # 记录每个行ID的来源
                        for row_id in row_ids:
                            row_to_source_mapping[row_id] = source_file
                    except Exception as e:
                        print(f"   ⚠️ 解析JSONL行时出错: {e}")
                        continue

            # 创建完整的映射数据结构
            mapping_data = {
                "batch_mapping": batch_mapping,
                "local_id_to_source": row_to_source_mapping,
                "task_id": task_id,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 保存映射到任务目录
            mapping_path = os.path.join(task_dir, "batch_mapping.json")
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            print(f"   ✓ 已创建批次映射文件: {mapping_path}")

            # 同时在全局映射目录创建副本，以便于跨任务访问
            global_copy_path = os.path.join(
                global_mapping_dir, f"batch_mapping_{task_id}.json"
            )
            with open(global_copy_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            print(f"   ✓ 已创建批次映射全局副本: {global_copy_path}")

        except Exception as e:
            print(f"   ⚠️ 创建批次映射文件失败: {e}")

        # 初始化OpenAI客户端
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # 上传JSONL文件
        print(f"   >> 上传输入文件...")
        with open(jsonl_path, "rb") as file:
            file_object = client.files.create(file=file, purpose="batch")
        print(f"   ✓ 文件上传成功! 文件ID: {file_object.id}")

        # 创建Batch任务
        print(f"   >> 创建Batch任务 (endpoint: {batch_endpoint})...")
        batch = client.batches.create(
            input_file_id=file_object.id,
            endpoint=batch_endpoint,
            completion_window=completion_window,
        )
        print(f"   ✅ Batch任务创建成功! 任务ID: {batch.id}")

        # 将batch_id保存到任务目录(方便后续查询)
        with open(os.path.join(task_dir, "batch_id.txt"), "w") as f:
            f.write(batch.id)

        # 创建批次ID到任务ID的全局映射文件
        global_batch_mapping_file = os.path.join("uploads", "global_batch_mapping.json")
        global_batch_mapping = {}

        # 如果文件存在，先读取现有映射
        if os.path.exists(global_batch_mapping_file):
            try:
                with open(global_batch_mapping_file, "r", encoding="utf-8") as f:
                    global_batch_mapping = json.load(f)
            except Exception as e:
                print(f"   ⚠️ 读取全局批次映射失败: {e}")

        # 更新映射：记录哪个批次ID对应哪个任务ID以及映射文件路径
        global_batch_mapping[batch.id] = {
            "task_id": task_id,
            "mapping_file": mapping_path,
            "global_mapping_file": global_copy_path,  # 添加全局映射文件路径
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 保存更新后的全局映射
        try:
            with open(global_batch_mapping_file, "w", encoding="utf-8") as f:
                json.dump(global_batch_mapping, f, ensure_ascii=False, indent=2)
            print(f"   ✓ 已更新全局批次映射: {global_batch_mapping_file}")
        except Exception as e:
            print(f"   ⚠️ 保存全局批次映射失败: {e}")

        return batch.id

    except Exception as e:
        print(f"   ❌ 提交百炼Batch任务失败: {e}")
        print(traceback.format_exc())
        raise


def check_bailian_job_status(batch_id: str, config: dict) -> dict:
    """
    检查阿里云百炼Batch任务状态。

    Args:
        batch_id: Batch任务ID。
        config: 包含配置信息的字典。

    Returns:
        dict: 包含任务状态信息的字典。
    """
    print(f"   >> 检查Batch任务状态 (ID: {batch_id})...")

    # 从嵌套结构中获取llm_config
    llm_config = config.get("llm_config", {})

    # 获取配置
    api_key = llm_config.get("DASHSCOPE_API_KEY", "")
    base_url = llm_config.get(
        "BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    try:
        # 初始化OpenAI客户端
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # 查询任务状态
        batch = client.batches.retrieve(batch_id=batch_id)
        print(f"   >> Batch任务状态: {batch.status}")

        return {
            "status": batch.status,
            "created_at": batch.created_at,
            "expires_at": batch.expires_at,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
        }

    except Exception as e:
        print(f"   ❌ 检查Batch任务状态失败: {e}")
        print(traceback.format_exc())
        raise


def download_and_process_bailian_results(
    batch_id: str, original_df: pd.DataFrame, task_id: str, config: dict
) -> pd.DataFrame:
    """
    下载并处理阿里云百炼Batch任务结果。
    支持多文件批次处理，结果按原文件分组并保留来源信息。

    Args:
        batch_id: Batch任务ID。
        original_df: 原始数据DataFrame，此参数将被忽略，不再需要进行合并
        task_id: 任务ID，使用时间戳格式(如task_20250506_174440)。
        config: 包含配置信息的字典。

    Returns:
        pd.DataFrame: 处理后的DataFrame，仅包含百炼API返回结果。
    """
    try:
        print(f"   >> 开始下载并处理百炼Batch任务结果...（仅处理API返回）")

        # 从嵌套结构中获取llm_config
        llm_config = config.get("llm_config", {})

        # 获取配置
        api_key = llm_config.get("DASHSCOPE_API_KEY", "")
        base_url = llm_config.get(
            "BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        target_columns_config = llm_config.get("TARGET_COLUMNS", [])

        print(f"   >> 使用API KEY: {api_key[:5]}...，BASE URL: {base_url}")

        # 验证API Key格式
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("缺少有效的阿里云百炼API密钥")

        # 创建任务目录
        task_dir = os.path.join("uploads", task_id)
        os.makedirs(task_dir, exist_ok=True)

        # 初始化本地行ID到批次行ID的映射
        batch_mappings = {}
        mapping_found = False

        # 尝试多个可能的映射文件位置
        mapping_files = [
            os.path.join("uploads", "global_batch_mapping.json"),
            os.path.join(task_dir, "batch_mapping.json"),
            os.path.join(task_dir, "file_0", "batch_mapping.json"),
        ]

        for mapping_file in mapping_files:
            if os.path.exists(mapping_file):
                try:
                    with open(mapping_file, "r", encoding="utf-8") as f:
                        batch_mappings = json.load(f)
                    print(f"   ✓ 从 {mapping_file} 加载了批次映射")
                    mapping_found = True
                    break
                except Exception as e:
                    print(f"   ⚠️ 读取批次映射文件 {mapping_file} 时出错: {e}")

        if not mapping_found:
            print(f"   ⚠️ 无法找到有效的批次映射，使用空映射继续...")

        # 获取当前批次的行ID列表
        batch_row_ids = []
        current_batch_mapping = batch_mappings.get(batch_id, {})

        if not current_batch_mapping:
            for batch_info in batch_mappings.values():
                if batch_info.get("batch_id") == batch_id:
                    current_batch_mapping = batch_info
                    print(f"   ✓ 找到批次 {batch_id} 的映射")
                    break

        if current_batch_mapping:
            batch_row_ids = current_batch_mapping.get("row_ids", [])
            print(f"   ✓ 找到 {len(batch_row_ids)} 个行ID与批次 {batch_id} 关联")
        else:
            print(f"   ⚠️ 警告: 找不到批次 {batch_id} 的行ID映射")

        # 初始化OpenAI客户端
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # 检查任务状态并获取文件ID
        batch = client.batches.retrieve(batch_id=batch_id)
        if batch.status != "completed":
            raise ValueError(f"Batch任务状态为 {batch.status}，而非 completed")

        # 获取结果文件ID
        output_file_id = batch.output_file_id
        error_file_id = getattr(batch, "error_file_id", None)

        # 下载结果文件
        print(f"   >> 下载结果文件 (ID: {output_file_id})...")
        output_content = client.files.content(output_file_id)
        output_path = os.path.join(task_dir, "batch_result.jsonl")
        output_content.write_to_file(output_path)
        print(f"   ✓ 结果文件已保存到: {output_path}")

        # 如果有错误文件，也下载
        if error_file_id:
            print(f"   >> 下载错误文件 (ID: {error_file_id})...")
            error_content = client.files.content(error_file_id)
            error_path = os.path.join(task_dir, "batch_error.jsonl")
            error_content.write_to_file(error_path)
            print(f"   ✓ 错误文件已保存到: {error_path}")

        # 解析结果文件
        print(f"   >> 开始解析结果文件...")
        llm_results = []

        with open(output_path, "r", encoding="utf-8") as f:
            line_count = 0
            for line in f:
                line_count += 1
                try:
                    result_item = json.loads(line)
                    llm_results.append(result_item)
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ 第 {line_count} 行解析错误: {e}")

        # 保存解析后的结果
        result_json_path = os.path.join(task_dir, "batch_result_parsed.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(llm_results, f, ensure_ascii=False, indent=2)

        print(f"   ✓ 成功解析 {len(llm_results)} 个结果项")

        # 处理解析后的结果 - 修复这部分逻辑以正确提取content中的业务数据
        results_df = pd.DataFrame()
        parsed_results = []

        # 处理返回的每个结果项
        matched_count = 0
        for i, result_item in enumerate(llm_results):
            # 打印第一个结果项示例
            if i == 0:
                print(
                    f"   >> 结果项示例: {json.dumps(result_item, ensure_ascii=False)[:200]}..."
                )

            try:
                # 从嵌套的JSON结构中提取content字段 - 这是百炼API的关键部分
                content = (
                    result_item.get("response", {})
                    .get("body", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                # 记录content内容示例
                if i == 0 and content:
                    print(f"   >> Content内容示例: {content[:200]}...")

                # content本身是JSON字符串，需要再次解析
                if content:
                    try:
                        # 尝试解析content中的JSON
                        content_data = json.loads(content)

                        # 记录解析后的数据结构
                        if i == 0:
                            print(
                                f"   >> 解析后的数据结构: {json.dumps(content_data, ensure_ascii=False)[:200]}..."
                            )

                        # content通常是一个列表，包含业务数据对象
                        if isinstance(content_data, list):
                            for item in content_data:
                                if not isinstance(item, dict):
                                    print(
                                        f"   >> 警告: content_data列表项不是字典，跳过此项"
                                    )
                                    continue

                                # 为每个项目添加必要的字段，但不再添加local_row_id
                                # item["local_row_id"] = (
                                #     batch_row_ids[i] if i < len(batch_row_ids) else ""
                                # )

                                # 保留原始来源字段值，这是关键 - 确保不修改文件名
                                if "来源" in item:
                                    source_value = item["来源"]
                                    if i < 3:  # 只记录前几个项的日志，避免日志过多
                                        print(f"   >> 保留原始来源值: '{source_value}'")
                                else:
                                    print(
                                        f"   >> 警告: 结果项中缺少'来源'字段，添加空值"
                                    )
                                    item["来源"] = ""

                                # 确保行ID存在，这是非常重要的唯一标识符
                                if "行ID" not in item or not item["行ID"]:
                                    print(
                                        f"   >> 警告: 结果项中缺少'行ID'字段，生成UUID"
                                    )
                                    item["行ID"] = str(uuid.uuid4())

                                # 添加飞书所需系统字段，但不再生成record_table字段
                                item["record_id"] = ""  # 留空，表示新记录
                                # 删除下面这行，不再添加record_table字段
                                # item["record_table"] = ""  # 留空，表示不和任何表关联

                                # 记录行ID
                                if i < 3:  # 只记录前几个项的日志
                                    print(f"   >> 使用行ID: '{item['行ID']}'")

                                # 收集解析结果
                                parsed_results.append(item)
                                matched_count += 1
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ 解析content字段JSON时出错: {e}")
                        print(f"   >> content内容: {content[:100]}...")
                        continue
            except Exception as e:
                print(f"   ⚠️ 解析结果项 {i} 时出错: {e}")
                traceback.print_exc()
                continue

        # 将所有结果项转换为DataFrame
        if parsed_results:
            results_df = pd.DataFrame(parsed_results)
            print(f"   ✓ 成功匹配 {matched_count}/{len(llm_results)} 个结果项")

            # 记录所有来源值的统计信息
            if "来源" in results_df.columns:
                source_values = results_df["来源"].value_counts().to_dict()
                print(f"   >> 来源字段统计: {source_values}")
            else:
                print("   >> 警告: 结果DataFrame中缺少'来源'列")

            # 保存处理结果
            results_csv_path = os.path.join(task_dir, "processed_results.csv")
            results_df.to_csv(results_csv_path, index=False)
            print(f"   ✓ 已保存处理结果到: {results_csv_path}")
            print(f"   >> 处理完成，共获取 {len(results_df)} 条结果。")
        else:
            print("   ⚠️ 警告: 未能从百炼API结果中提取有效数据")
            results_df = pd.DataFrame()

        return results_df

    except Exception as e:
        print(f"   ❌ 下载并处理百炼Batch任务结果失败: {e}")
        print(traceback.format_exc())
        raise


def extract_standardize_batch_with_llm(
    batch_rows: list[dict],  # 当前批次的数据行 (字典列表)
    source_headers: list,  # 源文件的列标题列表
    target_columns: list,  # 目标输出列名列表
    api_key: str,  # DeepSeek API Key
    api_endpoint: str,  # DeepSeek API 端点 URL
    max_tokens: int,  # API 调用允许的最大完成 token 数
    timeout: int,  # API 请求的超时时间 (秒)
) -> Union[List[Dict], None]:
    """
    使用 DeepSeek LLM API 对一批数据进行信息提取和标准化。
    包含重试逻辑和常见的 API 错误处理。

    Args:
        batch_rows: 当前批次的数据，每个元素是一个字典，代表一行。
        source_headers: 源 Excel/CSV 的列标题列表。
        target_columns: 需要提取和生成的标准列名列表。
        api_key: DeepSeek API 密钥。
        api_endpoint: DeepSeek API 的 URL。
        max_tokens: API 调用时 `max_tokens` 参数。
        timeout: API 请求的超时时间 (秒)。

    Returns:
        Union[List[Dict], None]: 成功处理则返回标准化后的数据列表 (可能包含错误标记)，
                           如果 API 调用失败或发生严重错误，则返回 None。
                           如果 API Key 无效，则返回包含 API_KEY_ERROR 标记的列表。
    """
    # 如果输入批次为空，直接返回空列表
    if not batch_rows:
        return []

    # 检查 API Key 是否有效
    if not api_key or not api_key.startswith("sk-"):
        print("⚠️ DeepSeek API Key 缺失或无效。跳过 LLM 处理。")
        # 为批次中的每一行返回一个错误标记字典
        return [{col: "API_KEY_ERROR" for col in target_columns} for _ in batch_rows]

    # --- 准备 API 请求 ---
    # 将目标列名列表转换为 JSON 字符串，用于 Prompt
    target_schema_json = json.dumps(target_columns, ensure_ascii=False)
    # 将源列标题转换为字符串列表，再转为 JSON 字符串
    source_headers_str = [str(h) for h in source_headers]
    source_headers_json = json.dumps(source_headers_str, ensure_ascii=False)

    # 构建 System Prompt (系统消息)
    system_content = f"""
你是一个专业的数据处理引擎。你的核心任务是从用户提供的源数据批次 (Source Batch Data) 中，为每一行数据提取信息，并严格按照指定的目标模式 (Target Schema) 进行标准化。
**重要：你的输出必须是一个有效的 JSON 数组 (列表)，数组中的每个元素都是一个对应输入行的、符合 Target Schema 的 JSON 对象。元素的数量必须与输入数组完全一致。**
**目标模式 (Target Schema):**
```json
{target_schema_json}
```
**当源数据中电话/手机号列包含多个号码时（可能用分号、逗号、空格等分隔），你必须将该行拆分为多行，每行对应一个手机号，其他信息保持一致。**
**当手机号为这个格式：133-0437-0109，请转化为正确格式：13304370109。**
**手机号分隔符识别规则：**
- 支持的分隔符包括：中英文分号（;；）、中英文逗号（,，）、空格、斜杠（/）
- 一行中可能包含多个不同类型的分隔符

**处理规则:**
1.独立处理: 独立分析 "Source Batch Data" 数组中的每一个 JSON 对象。
2.提取与映射: 结合 "Source Headers" 上下文，将信息映射到 "Target Schema" 字段。找不到信息则对应值设为 ""。所有值必须为字符串。
3.最终输出格式: 你的最终输出必须且只能是一个有效的 JSON 数组 (列表)。
4.数组内容: 此数组包含 {len(batch_rows)} 个元素，每个元素是符合 "Target Schema" 的 JSON 对象。
5.绝对禁止: 绝对不要将最终的 JSON 数组包装在任何其他 JSON 对象或键中（例如，不要像 {{ "results": [...] }} 或 {{ "processed_data": [...] }} 这样）。直接输出 [ 开头，] 结尾的数组本身。
6.无额外内容: 不要包含任何解释、注释或标记。
请严格按照要求，直接生成最终的 JSON 数组。
        """

    # 将当前批次的数据行转换为 JSON 字符串，确保所有值为字符串
    batch_data_json = json.dumps(
        [{k: str(v) for k, v in row.items()} for row in batch_rows],
        ensure_ascii=False,  # 允许非 ASCII 字符
        indent=None,  # 不进行缩进，减少 token 占用
    )

    print(f"source_headers_json: {source_headers_json}")
    print(f"batch_data_json: {batch_data_json}")
    # 构建 User Prompt (用户消息)
    user_content = f"""
        这是本次需要处理的源数据表格的列标题 (Source Headers):
        {source_headers_json}
        这是包含 {len(batch_rows)} 行源数据的 JSON 数组 (Source Batch Data):
        {batch_data_json}
        请根据你在 System 指令中被赋予的角色和规则，处理这个批次的数据，并返回标准化的 JSON 数组。
        """

    # 设置请求头
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    # 构建请求体 (payload)
    payload = {
        "model": "deepseek-chat",  # 可以考虑将其也设为可配置项
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 8192,
        "temperature": 0.5,  # 可以考虑设为可配置项，1 表示较高的创造性/随机性
        # "stream": False, # 流式输出目前不适用于批处理解析
    }

    # --- API 调用与错误处理 ---
    max_retries = 2  # 最大重试次数
    retry_delay = 5  # 重试间隔时间 (秒)

    # 循环尝试调用 API
    for attempt in range(max_retries):
        try:
            print(
                f"      ... 发送批次数据到 LLM API (尝试 {attempt + 1}/{max_retries})..."
            )
            # 发送 POST 请求
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,  # 使用传入的超时时间
            )
            # 检查 HTTP 状态码，如果不是 2xx，则抛出异常
            response.raise_for_status()
            # 解析返回的 JSON 数据
            result_json = response.json()
            # (调试用) 打印完整的返回 JSON
            # print(json.dumps(result_json, ensure_ascii=False, indent=2))

            content_str = None
            # 提取 API 使用情况 (token 数量)
            usage_info = result_json.get("usage")
            if usage_info:
                print(
                    f"      [API 使用情况]: Prompt: {usage_info.get('prompt_tokens', 'N/A')}, Completion: {usage_info.get('completion_tokens', 'N/A')}, Total: {usage_info.get('total_tokens', 'N/A')}"
                )

            # 从返回结果中提取模型生成的内容
            if "choices" in result_json and result_json["choices"]:
                message = result_json["choices"][0].get("message", {})
                content_str = message.get("content")
            print(f"content_str: {content_str}")
            # --- 解析和验证 LLM 返回的内容 ---
            if content_str:
                try:
                    # 清理返回内容两端的空白字符和可能的代码块标记
                    content_str_cleaned = content_str.strip()
                    if content_str_cleaned.startswith("```json"):
                        content_str_cleaned = content_str_cleaned[7:-3].strip()
                    elif content_str_cleaned.startswith("```"):
                        content_str_cleaned = content_str_cleaned[3:-3].strip()

                    # 尝试将清理后的字符串解析为 JSON (预期是一个列表)
                    standardized_batch = json.loads(content_str_cleaned)

                    # 验证返回的是否为列表，且列表长度大于等于输入批次长度（支持多手机号拆分）
                    if isinstance(standardized_batch, list) and len(
                        standardized_batch
                    ) >= len(batch_rows):
                        if len(standardized_batch) > len(batch_rows):
                            print(
                                f"      -> LLM返回了 {len(standardized_batch)} 行（原始输入 {len(batch_rows)} 行），可能包含手机号拆分后的多行数据。"
                            )

                        # 建立原始行索引到可能的拆分行映射关系
                        original_row_mapping = {}
                        processed_rows = 0

                        # 遍历LLM返回的每一行结果
                        for result_idx, result_row in enumerate(standardized_batch):
                            if isinstance(result_row, dict):
                                # 确定此行对应的原始行索引
                                # 对于拆分行，多个结果行会对应同一个原始行
                                original_idx = min(result_idx, len(batch_rows) - 1)

                                # 当处理到新的原始行时，增加计数
                                if original_idx not in original_row_mapping:
                                    original_row_mapping[original_idx] = 0

                                # 获取原始行中的行ID（如果存在）或生成新的行ID
                                if "行ID" in batch_rows[original_idx]:
                                    row_id = batch_rows[original_idx]["行ID"]
                                else:
                                    row_id = str(uuid.uuid4())

                                # 保存结果行信息，使用行ID而不是local_row_id
                                result_row["行ID"] = row_id
                                source_name = local_id_to_source_map.get(
                                    row_id, "UNKNOWN_SOURCE"
                                )
                                result_row[source_column_name] = source_name
                                processed_batch_with_ids_and_source.append(result_row)

                                # 增加这个原始行的处理计数
                                original_row_mapping[original_idx] += 1
                                processed_rows += 1
                            else:
                                # 处理非字典项
                                print(
                                    f"      ⚠️ 警告: LLM 返回列表项不是字典 (索引 {result_idx})。添加错误占位符。"
                                )
                                # 确定最相近的原始行索引
                                original_idx = min(result_idx, len(batch_rows) - 1)

                                # 获取原始行中的行ID（如果存在）或生成新的行ID
                                if "行ID" in batch_rows[original_idx]:
                                    row_id = batch_rows[original_idx]["行ID"]
                                else:
                                    row_id = str(uuid.uuid4())

                                error_row = {
                                    col: "LLM_ITEM_FORMAT_ERROR"
                                    for col in target_columns
                                }
                                error_row["行ID"] = row_id
                                error_row[source_column_name] = (
                                    local_id_to_source_map.get(row_id, "UNKNOWN_SOURCE")
                                )
                                processed_batch_with_ids_and_source.append(error_row)

                        # 打印拆分结果统计
                        split_stat = ", ".join(
                            [
                                f"行{idx+1}: {count}条"
                                for idx, count in original_row_mapping.items()
                            ]
                        )
                        print(f"      拆分统计：{split_stat}")

                        # Filter for valid results *after* adding ID and Source
                        valid_results_in_batch = [
                            res
                            for res in processed_batch_with_ids_and_source
                            if not any(
                                str(v).startswith(("LLM_", "API_KEY_"))
                                for k, v in res.items()
                                # Exclude 行ID and 来源 from error check
                                if k not in ["行ID", source_column_name]
                            )
                        ]
                        consolidated_data.extend(
                            processed_batch_with_ids_and_source
                        )  # Add results with ID and Source
                        rows_processed_in_file_success += len(valid_results_in_batch)
                        total_rows_successfully_processed += len(valid_results_in_batch)
                        print(
                            f"      批次 {batch_num+1} 处理完成。收到并添加 ID 和来源到 {len(processed_batch_with_ids_and_source)} 条结果 ({len(valid_results_in_batch)} 条有效)。"
                        )

                    else:
                        # Handle length mismatch
                        print(
                            f"      ❌ 警告: LLM 返回列表长度不匹配... 添加错误标记。"
                        )
                        for local_id in batch_rows:
                            error_row = {
                                col: "BATCH_LENGTH_MISMATCH" for col in target_columns
                            }
                            error_row["行ID"] = local_id
                            error_row[source_column_name] = local_id_to_source_map.get(
                                local_id, "UNKNOWN_SOURCE"
                            )
                            consolidated_data.append(error_row)
                except json.JSONDecodeError as json_err:
                    # 如果返回的内容无法解析为 JSON
                    print(
                        f"      ❌ LLM 返回内容不是有效的 JSON 数组 (尝试 {attempt + 1}): {json_err}"
                    )
                    # 打印部分原始返回内容，帮助诊断
                    print(
                        f"         原始返回内容 (前 500 字符): {content_str[:500]}..."
                    )
                    # 进入重试流程
                except Exception as e:
                    # 捕获解析过程中其他未预料的错误
                    print(f"      ❌ 解析 LLM 返回的 JSON 数组时出错: {e}")
                    print(traceback.format_exc())
                    # 遇到未知解析错误，不再重试此批次
                    break

            else:
                # 如果 API 返回结果中没有 'content'
                print(f"      ❌ LLM 返回结果缺少 'content' (尝试 {attempt + 1})。")
                if "error" in result_json:
                    print(f"         API 错误信息: {result_json['error']}")
                # 进入重试流程

        # --- 捕获网络和 API 相关的异常 ---
        except requests.exceptions.Timeout:
            print(f"      ❌ LLM API 请求超时 (尝试 {attempt + 1})。")
            # 进入重试流程
        except requests.exceptions.HTTPError as http_err:
            # 捕获 HTTP 错误 (如 4xx, 5xx)
            print(f"      ❌ LLM API HTTP 错误 (尝试 {attempt + 1}): {http_err}")
            if http_err.response is not None:
                status_code = http_err.response.status_code
                response_text = http_err.response.text
                print(f"         状态码: {status_code}")
                # 特殊处理：上下文长度超限错误 (400 Bad Request)
                if status_code == 400 and (
                    "context_length_exceeded" in response_text.lower()
                    or "prompt is too long" in response_text.lower()
                    or "maximum context length" in response_text.lower()
                ):
                    print(
                        "      ❌ 错误: 输入内容可能超过模型上下文长度限制! 请减小 BATCH_SIZE。"
                    )
                    # 抛出 ValueError，这将终止整个任务的处理
                    raise ValueError("上下文长度超限")
                # 特殊处理：认证/授权错误 (401, 403)
                elif status_code in [401, 403]:
                    print(
                        f"      ❌ API 认证/授权错误 ({status_code})。请检查 API Key。"
                    )
                    raise ValueError("API 认证/授权错误")
                # 特殊处理：速率限制错误 (429)
                elif status_code == 429:
                    print(f"      ❌ 达到 API 速率限制 ({status_code})。")
                    # 如果还有重试次数，则等待更长时间后重试
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 2)  # 增加等待时间
                        print(f"         将在 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue  # 继续下一次循环尝试
                    else:
                        # 重试次数用尽
                        print("      ❌ 重试后仍然达到速率限制。")
                        raise ValueError("API 速率限制")
                else:
                    # 其他 HTTP 错误
                    print(f"         响应内容 (前 500 字符): {response_text[:500]}...")
            else:
                print("      ❌ HTTP 错误但没有响应对象。")
            # 对于 HTTP 错误 (非 429 且非致命错误)，进入重试流程 (如果还有次数)
        except requests.exceptions.RequestException as e:
            # 捕获其他网络请求相关的错误 (如 DNS 解析失败、连接错误等)
            print(f"      ❌ LLM API 请求失败 (尝试 {attempt + 1}): {e}")
            # 进入重试流程
        except Exception as e:
            # 捕获调用 API 过程中的其他未知错误
            print(f"      ❌ 调用 LLM API 时发生未知错误 (批处理): {e}")
            print(traceback.format_exc())
            # 重新抛出未知错误，终止处理
            raise e

        # --- 重试逻辑 ---
        # 如果当前尝试失败且还有重试次数
        if attempt < max_retries - 1:
            print(f"      将在 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        else:
            # 重试次数已用完
            print(f"      ❌ 此批次已达到最大重试次数。")

    # 如果所有重试都失败了
    print(f"      ⚠️ 处理失败，为此批次返回 None。")
    return None  # 返回 None 表示此批次处理失败


# === 主处理函数 ===
def process_files_and_consolidate(
    input_files: List[str],  # 输入文件路径列表
    output_file_path: str,  # 输出 Excel 文件路径
    config: dict,  # 包含配置项的字典 (API Key, Target Columns 等)
    update_progress_callback=None,  # 用于报告进度的回调函数 (可选)
) -> Union[str, Tuple[str, str]]:
    """
    核心处理流程函数，由 app.py 在后台线程中调用。
    修改为使用阿里云百炼 Batch API 处理。

    新流程:
    1. 遍历输入文件列表并逐个处理，不合并文件
    2. 为每个文件单独创建批次，添加到统一的JSONL文件
    3. 提交给百炼 Batch API 进行处理
    4. 返回 batch_id 供 app.py 后续轮询

    Args:
        input_files: 包含一个或多个输入文件完整路径的列表。
        output_file_path: 要保存的最终 Excel 文件的完整路径。
        config: 包含所有配置的字典。
        update_progress_callback: 一个函数，接受 (状态消息, 进度百分比, 已处理文件数, 总文件数) 参数。

    Returns:
        Union[str, Tuple[str, str]]:
            - 当使用阿里云百炼 API 时，返回 (batch_id, task_id) 元组
            - 当处理完成时，返回输出文件的路径。
    """

    # --- 1. 解包配置项 ---
    api_mode = "bailian"  # 默认使用阿里云百炼 API

    # 获取llm_config中的配置（处理嵌套结构）
    llm_config = config.get("llm_config", {})
    print(f"config详情: {json.dumps(config, ensure_ascii=False)}")

    # 从llm_config中获取百炼API相关配置
    dashscope_api_key = llm_config.get("DASHSCOPE_API_KEY", "")

    # 检查百炼API相关配置
    print(f"百炼API配置: DASHSCOPE_API_KEY存在: {bool(dashscope_api_key)}")
    print(
        f"百炼API配置: DASHSCOPE_API_KEY格式正确: {bool(dashscope_api_key and dashscope_api_key.startswith('sk-'))}"
    )
    print(
        f"百炼API配置: BAILIAN_MODEL_NAME: {llm_config.get('BAILIAN_MODEL_NAME', '未配置')}"
    )
    print(
        f"百炼API配置: BAILIAN_COMPLETION_WINDOW: {llm_config.get('BAILIAN_COMPLETION_WINDOW', '未配置')}"
    )

    if not dashscope_api_key or not dashscope_api_key.startswith("sk-"):
        api_mode = "deepseek"  # 回退到 DeepSeek API
        print(f"   ⚠️ 警告: 未找到有效的阿里云百炼 API Key，将使用 DeepSeek API。")

    # 生成基于时间戳的任务ID作为本次处理的唯一标识
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_id = f"task_{timestamp}"
    print(f"--- 开始处理，任务ID: {task_id} ---")

    # --- 2. 检查输入文件 ---
    if not input_files:
        print("❌ 错误: 未提供任何输入文件。")
        if update_progress_callback:
            update_progress_callback("错误: 未提供输入文件", 100, 0, 0)
        raise ValueError("未提供任何输入文件")

    total_files = len(input_files)
    print(f"   📄 处理 {total_files} 个输入文件...")

    # 创建任务目录
    task_dir = os.path.join("uploads", task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 保存任务信息到文件
    task_info = {
        "task_id": task_id,
        "input_files": [os.path.basename(f) for f in input_files],
        "output_file": output_file_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_mode": api_mode,
        "file_count": total_files,
    }
    with open(os.path.join(task_dir, "task_info.json"), "w") as f:
        json.dump(task_info, f, ensure_ascii=False, indent=2)

    # --- 3. 处理每个文件并创建批次 ---
    if api_mode == "bailian":
        print(f"   >> 使用阿里云百炼 Batch API 处理数据...")

        # 创建批次ID到行ID的全局映射
        global_batch_mapping = {}

        # 创建local_id到源文件名的映射
        local_id_to_source_map = {}

        # 创建统一的JSONL文件，用于保存所有文件的批次
        jsonl_path = os.path.join(task_dir, "batch_input.jsonl")

        # 记录总处理行数
        total_rows_processed = 0

        # 从llm_config中获取批处理相关配置
        batch_size = llm_config.get("BATCH_SIZE", 100)
        model_name = llm_config.get("BAILIAN_MODEL_NAME", "qwen-turbo-latest")
        batch_endpoint = llm_config.get(
            "BAILIAN_BATCH_ENDPOINT", "/v1/chat/completions"
        )
        max_tokens = llm_config.get("MAX_COMPLETION_TOKENS", 8192)

        # 准备目标列配置
        target_columns_config = llm_config.get(
            "TARGET_COLUMNS", ["公司名称", "联系人", "职位", "电话", "来源"]
        )

        llm_target_columns = target_columns_config

        # 构建系统提示词(system prompt)
        target_schema_json = json.dumps(llm_target_columns, ensure_ascii=False)
        system_content = f"""
你是一个专业的数据处理引擎。你的核心任务是从用户提供的源数据批次 (Source Batch Data) 中，为每一行数据自动推断并提取与目标模式 (Target Schema) 字段最匹配的信息，并严格按照指定的目标模式进行标准化。
**重要要求：**
1. 你的输出必须是一个有效的 JSON 数组 (列表)，数组中的每个元素都是一个对应输入行的、符合 Target Schema 的 JSON 对象。元素的数量必须与输入数组完全一致，除非去重规则生效。
2. 每个结果对象必须包含"来源"字段，且其值取自输入的来源表格名称。
3. 返回的所有列标题（字段名）必须与 Target Schema 完全一致、顺序一致，不允许多余或缺失字段。
4. 找不到的字段请置空字符串。

**目标模式 (Target Schema):**
```json
{target_schema_json}
```
**核心处理规则：**
**1. 电话号码规范化与格式转换：**
    - 原始电话号码中的所有非数字字符（例如：连字符 "-", 空格, 括号 "()" 等）都必须被去除，只保留纯数字序列。
    - 例如，如果输入为 "133-0437-0109"，则输出应为 "13304370109"。

**2. 企业名称清洗规则：**
   - 只保留企业名称的核心主体部分。
   - 必须去除所有附加描述性信息，例如：括号内的注释（如"（广东省500强）"）、注册资本信息（如"注册资本 1000万美元"）、荣誉称号、规模描述、地理位置描述等。
   - **正例**：
     - 企业名称输入："广州立邦涂料有限公司，（广东省500强）注册资本 1000万美元"
     - 企业名称输出："广州立邦涂料有限公司"
   - **反例**：
     - 错误输出："广州立邦涂料有限公司，（广东省500强）注册资本 1000万美元"

**3. 多行拆分与混杂内容处理规则：**
   - **目标**：如果单个输入行包含多个独立的联系人信息单元，则必须将该行拆分为多个输出行，每个输出行对应一个联系人信息单元。
   - **识别混杂内容**：仔细检查所有输入字段（尤其是"职务"、"备注"等自由文本字段）以识别独立的联系人实体。一个联系人实体通常由姓名、电话号码和/或职位信息组成。
   - **拆分逻辑**：
     - 为每个从混杂内容中识别出的独立联系人实体生成一条新的输出行。
     - 如果"电话"或"手机号"列明确包含由分隔符（支持的分隔符包括：中英文分号（;；）、中英文逗号（,，）、空格、斜杠（/））分隔的多个号码，并且没有更具体的联系人信息与之对应，也应为每个号码生成一条新的输出行。
     - 当"混杂内容处理规则"识别出多个完整联系人实体时，其拆分逻辑优先。
   - **信息填充**：对于每个生成的输出行，应填充从混杂内容中提取出的对应实体的姓名、电话和职位到 Target Schema 的相应字段。
   - **输入示例** (处理混杂内容)：
     ```json
    {{ // 注意: f-string 中的 JSON 示例，花括号需要转义
    "序号": 83691,
    "公司名称": "广州仕邦人力资源有限公司",
    "职务": "18520784668  IT 谢生（昀昌）    020-66393586 财务部     财务总监为张生，",
    "姓名": "",
    "电话": "",
    "行ID": "c3b4e24d-4d31-4102-aac2-0fbbb8825e7d",
    "来源": "特殊场景测试2（电话联系人在一列）.xlsx"
    }}
     ```
   - **输出示例** (对应上述输入，生成两条记录)：
     ```json
     [
       {{ // 注意: f-string 中的 JSON 示例，花括号需要转义
        "公司名称": "广州仕邦人力资源有限公司",
        "姓名": "谢生（昀昌）",
        "电话": "18520784668", // 规范化后
        "职位": "IT",
        "行ID": "c3b4e24d-4d31-4102-aac2-0fbbb8825e7d",
        "来源": "特殊场景测试2（电话联系人在一列）.xlsx"
        }},
       {{ // 注意: f-string 中的 JSON 示例，花括号需要转义
        "公司名称": "广州仕邦人力资源有限公司",
        "姓名": "张生",
        "电话": "02066393586", // 规范化后
        "职位": "财务总监",
        "行ID": "c3b4e24d-4d31-4102-aac2-0fbbb8825e7d",
        "来源": "特殊场景测试2（电话联系人在一列）.xlsx"
        }}
     ]
     ```
**4. 基于"企业名称"和"电话"的输出去重规则：**
   - **目标**：确保在你最终生成的JSON数组中，对于拥有相同"企业名称"和规范化后"电话号码"的组合，只包含一条唯一的记录。
   - **处理逻辑**：当你处理输入数据并生成结果对象时，如果一个结果对象其"企业名称"和规范化后的"电话号码"与你在此次任务中*已经生成并计划包含在最终输出数组中*的某个对象完全相同，则应避免再次添加这个重复的对象。对于一组重复记录，请只保留你遇到的**第一条有效记录**。
   - **关键字段**：去重基于"企业名称"和"电话"这两个字段的组合。电话号码在比较前必须是经过规范化（仅含数字，已通过规则1处理）的。
   - **示例**：
     - 假设原始输入经过初步处理（包括拆分、规范化）后，可能产生如下中间结果，准备合并到最终输出：
       ```
       [
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "郭间荷", "职位": "国家税务总局佛山市南海区税务局", "电话": "15015608915", "行ID": "84167", "来源": "file.xlsx"}},
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "张妍", "职位": "", "电话": "86222923", "行ID": "84167", "来源": "file.xlsx"}}, // 不同电话，保留
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "刘永雄", "职位": "", "电话": "13802628974", "行ID": "84168", "来源": "file.xlsx"}},
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "郭间荷", "职位": "国家税务总局佛山市南海区税务局", "电话": "15015608915", "行ID": "84169", "来源": "file.xlsx"}}, // 与第一条的企业名称和电话重复
         {{"企业名称": "广东世纪达建设集团有限公司", "联系人": "刘永雄", "职位": "", "电话": "15015608915", "行ID": "84170", "来源": "file.xlsx"}}  // 与第一条的企业名称和电话重复，但联系人不同
       ]
       ```
     - **期望的去重后输出（单个JSON数组）**：
       ```json
       [
         {{
           "企业名称": "广东世纪达建设集团有限公司",
           "联系人": "郭间荷",
           "职位": "国家税务总局佛山市南海区税务局",
           "电话": "15015608915", // 规范化后
           "行ID": "84167",
           "来源": "file.xlsx"
         }},
         {{
           "企业名称": "广东世纪达建设集团有限公司",
           "联系人": "张妍",
           "职位": "",
           "电话": "86222923", // 规范化后
           "行ID": "84167", // 假设行ID允许在拆分或关联时复用
           "来源": "file.xlsx"
         }},
         {{
           "企业名称": "广东世纪达建设集团有限公司",
           "联系人": "刘永雄",
           "职位": "",
           "电话": "13802628974", // 规范化后
           "行ID": "84168",
           "来源": "file.xlsx"
         }}
         // 后续两条与第一条"企业名称"和"电话"重复的记录被去除，即使联系人字段可能不同，因为我们优先保留第一条。
       ]
       ```
   - **上下文**：此去重逻辑应用于你处理当前整个任务（即一个完整的源数据批次 Source Batch Data）所产生的所有记录，目标是确保最终返回的JSON数组内不含基于（企业名称，电话）的重复项。

**通用处理指南：**
1.  **自动推断与映射**：结合源数据内容，自动推断并提取与 Target Schema 字段最匹配的信息。
2.  **字段严格匹配**：只输出 Target Schema 中定义的字段，输出顺序与 Target Schema 保持一致，不要输出任何多余字段。
3.  **格式整理**：去除字段值中不必要的首尾空格。对于文本内容，避免去除关键的内部标点，除非特定规则要求（如电话号码规范化）。
4.  **批量处理意识**：你将收到一个包含多行数据的数组，你需要独立处理每一行（可能将其拆分为多行或根据去重规则舍弃），并将所有最终生成的记录合并到一个JSON数组中返回。
5.  **最终输出格式**：只返回一个顶级的JSON数组，数组内包含所有处理后的记录对象。绝对不要在JSON数组之外添加任何额外的解释、注释、markdown标记或任何非JSON文本。
"""

        # 开始逐个处理文件
        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for file_index, file_path in enumerate(input_files):
                current_file_num = file_index + 1
                file_basename = os.path.basename(file_path)
                print(
                    f"\n📄 处理文件 {current_file_num}/{total_files}: {file_basename}"
                )

                # 报告进度
                if update_progress_callback:
                    progress_pct = int(
                        (file_index / total_files) * 60
                    )  # 文件处理占总进度的60%
                    update_progress_callback(
                        f"处理文件 {current_file_num}/{total_files}: {file_basename}",
                        progress_pct,
                        file_index,
                        total_files,
                    )

                # 读取当前文件
                df_source = read_input_file(file_path)

                # 文件读取失败或为空时跳过
                if df_source is None or df_source.empty:
                    print(f"   ⚠️ 跳过文件 {file_basename} (读取失败或为空)。")
                    continue

                # 添加源文件名称到数据中
                df_source["来源"] = file_basename

                # 为每个行ID建立到源文件名的映射
                for row_id in df_source["行ID"].tolist():
                    local_id_to_source_map[row_id] = file_basename
                print(
                    f"   >> 已添加 {len(df_source)} 个行ID到来源 '{file_basename}' 的映射"
                )

                # 保存原始文件数据，用于后续处理
                file_dir = os.path.join(task_dir, f"file_{file_index}")
                os.makedirs(file_dir, exist_ok=True)
                file_df_path = os.path.join(file_dir, "original.csv")
                df_source.to_csv(file_df_path, index=False)

                # 获取该文件的总行数并按批次划分
                file_rows = len(df_source)
                file_batch_count = (
                    file_rows + batch_size - 1
                ) // batch_size  # 向上取整
                print(
                    f"   >> 文件 {file_basename} 共 {file_rows} 行，划分为 {file_batch_count} 个批次"
                )

                # 为此文件创建批次
                for batch_idx in range(file_batch_count):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, file_rows)
                    batch_df = df_source.iloc[start_idx:end_idx]

                    # 获取批次中所有行的行ID
                    batch_row_ids = batch_df["行ID"].tolist()

                    # 创建批次ID，确保全局唯一（添加文件索引作为前缀）
                    batch_custom_id = (
                        f"file{file_index}_batch_{batch_idx}_{len(batch_row_ids)}"
                    )

                    # 保存批次行ID映射
                    global_batch_mapping[batch_custom_id] = batch_row_ids

                    # 将批次数据转换为文本格式
                    batch_content = []
                    for _, row in batch_df.iterrows():
                        row_str = "\n".join(
                            [f"{k}: {v}" for k, v in row.items() if k != "行ID"]
                        )
                        # 添加行ID作为行标识
                        row_str += f"\n行ID: {row['行ID']}"
                        batch_content.append(row_str)

                    # 合并所有行内容
                    all_rows_content = "\n\n--- 行分隔符 ---\n\n".join(batch_content)

                    # 获取源列标题
                    source_headers = batch_df.columns.astype(str).tolist()
                    source_headers_json = json.dumps(source_headers, ensure_ascii=False)

                    # 获取源文件名
                    if "source_file" in batch_df.columns:
                        source_filename = batch_df["source_file"].iloc[0]
                    else:
                        # 尝试从batch_row_ids获取原始文件名
                        source_filename = local_id_to_source_map.get(
                            batch_row_ids[0], "未知来源"
                        )

                    # 记录使用的源文件名
                    print(
                        f"   >>> 批次 {batch_idx+1} 使用的源文件名: {source_filename}"
                    )

                    # 用户提示词(user prompt)
                    batch_dict_list = batch_df.to_dict(orient="records")
                    batch_data_json = json.dumps(batch_dict_list, ensure_ascii=False)
                    user_content = f"本批次数据来源于:{source_filename}\n本批次需要处理的数据:{batch_data_json}"

                    # 构建消息数组
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ]

                    # 构建API请求体
                    body = {
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.5,
                    }

                    # 构建JSONL记录
                    record = {
                        "custom_id": batch_custom_id,
                        "method": "POST",
                        "url": batch_endpoint,
                        "body": body,
                        "row_ids": batch_row_ids,
                        "source_file": file_basename,  # 添加源文件信息
                        "file_index": file_index,  # 添加文件索引信息
                    }

                    # 写入到统一的JSONL文件
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_rows_processed += file_rows
                print(f"   ✓ 已为文件 {file_basename} 创建 {file_batch_count} 个批次")

        # 保存全局批次映射到文件
        mapping_path = os.path.join(task_dir, "batch_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            # 添加额外的元数据到映射文件
            mapping_data = {
                "batch_mapping": global_batch_mapping,
                "files": [os.path.basename(f) for f in input_files],
                "total_rows": total_rows_processed,
                "batch_size": batch_size,
                "local_id_to_source": local_id_to_source_map,  # 添加local_id到源文件名的映射
            }
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        print(f"   ✓ 已保存全局批次映射到: {mapping_path}")

        # 保存原始文件列表
        with open(os.path.join(task_dir, "input_files.json"), "w") as f:
            json.dump(
                {
                    "files": [os.path.basename(f) for f in input_files],
                    "file_paths": input_files,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"   ✅ 已成功创建百炼Batch API输入文件: {jsonl_path} (共 {total_rows_processed} 行数据)"
        )

        if update_progress_callback:
            update_progress_callback(
                "提交百炼Batch API任务...",
                70,
                total_files,
                total_files,
            )

        # 统一提交百炼Batch任务
        batch_id = submit_bailian_batch_job(jsonl_path, task_id, config)

        if update_progress_callback:
            update_progress_callback(
                f"已提交百炼Batch任务(ID: {batch_id})，等待处理...",
                80,
                total_files,
                total_files,
            )

        # 返回batch_id和task_id，供app.py后续轮询
        return (batch_id, task_id)
    else:
        # 保留原有的DeepSeek API处理逻辑，用于备选
        print("   ⚠️ 使用原有DeepSeek API处理未实现。请先配置阿里云百炼API。")
        if update_progress_callback:
            update_progress_callback(
                "错误: DeepSeek API处理未实现，请配置阿里云百炼API",
                100,
                total_files,
                total_files,
            )
        raise NotImplementedError("当前版本仅支持阿里云百炼API处理")


# === 用于直接测试 processor.py 的示例代码 (注释掉) ===
# def print_progress(status, progress, files_done, total_files):
#     print(f"[进度更新] 状态: {status} | 进度: {progress}% | 文件: {files_done}/{total_files}")

# if __name__ == '__main__':
#     # 用于直接运行此脚本进行测试的虚拟配置
#     test_config = {
#         "TARGET_COLUMNS": ["公司名称", "姓名", "职务", "电话"],
#         "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "YOUR_DUMMY_KEY_FOR_TESTING"), # 从环境变量获取或使用虚拟 Key
#         "BATCH_SIZE": 5, # 使用较小的批处理大小进行测试
#         "DEEPSEEK_API_ENDPOINT": "https://api.deepseek.com/chat/completions",
#         "MAX_COMPLETION_TOKENS": 2048,
#         "API_TIMEOUT": 60, # 测试用较短超时
#     }
#     # 指定测试用的输入文件列表 (你需要创建这些文件或修改路径)
#     test_input_files = ['./test_data/input1.xlsx', './test_data/input2.csv']
#     # 指定测试用的输出文件路径
#     test_output = './test_output/consolidated_test_output.xlsx'

#     # 确保测试输出目录存在
#     os.makedirs(os.path.dirname(test_output), exist_ok=True)

#     print(f"--- 开始直接测试 processor.py ---")
#     print(f"测试输入文件: {test_input_files}")
#     print(f"测试输出文件: {test_output}")

#     try:
#         # 调用主处理函数进行测试，并传入打印进度的回调函数
#         result = process_files_and_consolidate(test_input_files, test_output, test_config, print_progress)
#         print(f"--- 测试完成，结果保存在: {result} ---")
#     except Exception as e:
#         print(f"--- 测试过程中发生错误 ---: {e}")
#         print(traceback.format_exc())
