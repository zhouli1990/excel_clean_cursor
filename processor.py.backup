# -*- coding: utf-8 -*-
import requests
import pandas as pd
import time
import numpy as np
import os
import json
import traceback  # 用于打印详细的错误堆栈信息
import math  # 用于计算批处理数量
import uuid  # Added import
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

# === Logger Setup ===
# Create a logger instance
logger = logging.getLogger(__name__)
# Set the logging level (e.g., INFO, DEBUG, WARNING)
logger.setLevel(logging.INFO)
# Create a handler (e.g., StreamHandler to output to console)
if not logger.handlers:  # Avoid adding multiple handlers if reloaded
    handler = logging.StreamHandler()
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(handler)

# === 配置项 (这些值现在由 app.py 传入) ===
# DEEPSEEK_API_KEY = "YOUR_API_KEY"       # 从 app.py 获取
# TARGET_COLUMNS = [...]                  # 从 app.py 获取
# BATCH_SIZE = 160                        # 从 app.py 获取
# MAX_COMPLETION_TOKENS = 8192            # 从 app.py 获取
# DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/chat/completions" # 从 app.py 获取


def read_input_file(
    file_path: str,  # required_columns: Optional[List[str]] = None # Removed this parameter
) -> pd.DataFrame:
    """
    Reads an Excel or CSV file into a Pandas DataFrame.
    # Removed validation of required columns.
    Adds a unique local_row_id to each row.
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
            df = pd.read_excel(file_path, engine="openpyxl", converters=converters)
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

        # Add local_row_id
        if "local_row_id" not in df.columns:
            df["local_row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
            logger.info(f"Added 'local_row_id' column to DataFrame from {file_path}.")
        else:
            # Handle case where column might exist but contain NaNs or duplicates
            existing_ids = df["local_row_id"].dropna().astype(str)
            if len(existing_ids) != len(df) or existing_ids.duplicated().any():
                logger.warning(
                    f"'local_row_id' column exists in {file_path} but contains nulls or duplicates. Regenerating IDs."
                )
                df["local_row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # *** REMOVED adding '来源' column here ***
        # df["来源"] = os.path.basename(file_path)

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
        "max_tokens": max_tokens,
        "temperature": 1,  # 可以考虑设为可配置项，1 表示较高的创造性/随机性
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
                        validated_batch = []
                        # 进一步验证列表中的每个元素是否为字典，并按目标列格式化
                        for i, item in enumerate(standardized_batch):
                            if isinstance(item, dict):
                                # 确保所有目标列都存在，且值为字符串，缺失值用空字符串填充
                                validated_item = {
                                    col: str(item.get(col, ""))
                                    for col in target_columns
                                }
                                validated_batch.append(validated_item)
                            else:
                                # 如果某一项不是字典，记录错误
                                print(
                                    f"      ⚠️ LLM 返回结果的第 {i+1} 项不是字典: {item}"
                                )
                                validated_batch.append(
                                    {
                                        col: "LLM_ITEM_FORMAT_ERROR"  # 标记格式错误
                                        for col in target_columns
                                    }
                                )
                        print(f"      -> LLM 批处理成功 ({len(validated_batch)} 行)。")
                        return validated_batch  # 成功处理，返回结果
                    else:
                        # 如果返回的不是列表或长度不匹配
                        details = f"类型: {type(standardized_batch).__name__}" + (
                            f", 长度: {len(standardized_batch)}"
                            if isinstance(standardized_batch, list)
                            else ""
                        )
                        print(
                            f"      ❌ LLM 返回了无效列表或长度异常 (预期至少 {len(batch_rows)} 行)。实际 -> {details} (尝试 {attempt + 1})。"
                        )
                        # 进入重试流程 (如果还有重试次数)

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
):
    """
    核心处理流程函数，由 app.py 在后台线程中调用。
    1. 遍历输入文件列表。
    2. 读取每个文件。
    3. 将文件内容分批。
    4. 调用 LLM API 处理每个批次。
    5. 合并处理结果。
    6. 将最终结果保存到 Excel 文件。
    7. 通过回调函数报告进度。

    Args:
        input_files: 包含一个或多个输入文件完整路径的列表。
        output_file_path: 要保存的最终 Excel 文件的完整路径。
        config: 包含所需配置的字典，如 TARGET_COLUMNS, DEEPSEEK_API_KEY 等。
        update_progress_callback: 一个函数，接受 (状态消息, 进度百分比, 已处理文件数, 总文件数) 参数。
                                  用于向 app.py 报告进度。

    Returns:
        str: 成功处理时返回输出文件的路径。

    Raises:
        ValueError: 如果没有输入文件、API 认证失败、上下文超限或最终没有提取到任何数据。
        Exception: 其他未处理的异常 (如文件写入失败)。
    """
    # --- 1. 解包配置项 ---
    target_columns_config = config.get(
        "TARGET_COLUMNS",
        ["公司名称", "联系人", "职位", "电话", "来源"],  # Keep '来源' in config default
    )
    source_column_name = "来源"

    # *** 修改: 准备给 LLM 的目标列 (不含来源) ***
    llm_target_columns = [
        col for col in target_columns_config if col != source_column_name
    ]
    # *** 修改: 最终输出列 (包含来源和 local_row_id) ***
    final_output_columns = target_columns_config + ["local_row_id"]
    # Ensure local_row_id is last and deduplicate if needed
    final_output_columns = list(dict.fromkeys(final_output_columns))
    if "local_row_id" in final_output_columns:
        final_output_columns.remove("local_row_id")
    final_output_columns.append("local_row_id")

    api_key = config.get("DEEPSEEK_API_KEY", "")
    api_endpoint = config.get(
        "DEEPSEEK_API_ENDPOINT", "https://api.deepseek.com/chat/completions"
    )
    batch_size = config.get("BATCH_SIZE", 160)
    max_tokens = config.get("MAX_COMPLETION_TOKENS", 8192)
    api_timeout = config.get("API_TIMEOUT", 180)

    print(f"--- 开始处理，批处理大小: {batch_size} ---")
    print(f"   LLM 目标列 (不含来源): {llm_target_columns}")
    print(f"   最终输出列: {final_output_columns}")

    consolidated_data = []
    total_rows_attempted = 0
    total_rows_successfully_processed = 0
    files_processed_count = 0
    files_with_errors = []
    total_files = len(input_files)
    # *** 新增：用于存储 local_id 到来源文件名的映射 ***
    local_id_to_source_map = {}

    # --- 2. 检查输入文件 ---
    if not input_files:
        print("❌ 错误: 未提供任何输入文件。")
        # 如果有回调函数，报告错误并设置进度为 100% (表示结束)
        if update_progress_callback:
            update_progress_callback("错误: 未提供输入文件", 100, 0, 0)
        # 抛出异常，终止处理
        raise ValueError("未提供任何输入文件")

    # --- 3. 循环处理每个文件 ---
    for i, file_path in enumerate(input_files):
        current_file_num = i + 1
        file_basename = os.path.basename(file_path)
        print(f"\n📄 处理文件 {current_file_num}/{total_files}: {file_basename}")
        # 报告开始处理当前文件
        if update_progress_callback:
            # 进度基于已开始处理的文件数量计算
            progress_pct = int((i / total_files) * 100)
            update_progress_callback(
                f"读取文件 {current_file_num}/{total_files}: {file_basename}",
                progress_pct,
                i,  # 已完成的文件数 (从 0 开始)
                total_files,
            )

        # --- 3.1 读取文件 ---
        df_source = read_input_file(file_path)

        # 如果文件读取失败 (返回 None)
        if df_source is None or df_source.empty:
            print(f"   ⚠️ 跳过文件 {file_basename} (读取失败)。")
            files_with_errors.append(file_basename)
            # 继续处理下一个文件
            continue
        # 如果文件为空
        if df_source.empty:
            print(f"   ℹ️ 文件 {file_basename} 为空。跳过。")
            # 即使为空也算作已处理 (因为它被成功读取了)
            # files_processed_count += 1 # 根据需要决定是否将空文件计入已处理
            continue

        # 文件成功读取且不为空
        files_processed_count += 1
        num_rows_in_file = len(df_source)  # 当前文件行数
        print(f"   文件包含 {num_rows_in_file} 行。")

        # *** 新增：填充 local_id 到来源的映射 ***
        for local_id in df_source["local_row_id"].tolist():
            local_id_to_source_map[local_id] = file_basename
        print(
            f"   已记录 {len(df_source)} 个 local_id 到来源 '{file_basename}' 的映射。"
        )

        # Get source headers
        source_headers = df_source.columns.astype(str).tolist()
        rows_processed_in_file_success = 0
        num_batches = math.ceil(num_rows_in_file / batch_size)

        # --- 3.2 分批处理文件内容 ---
        for batch_num, batch_start_index in enumerate(
            range(0, num_rows_in_file, batch_size)  # 按 batch_size 步长生成起始索引
        ):
            # 计算当前批次的结束索引
            batch_end_index = min(batch_start_index + batch_size, num_rows_in_file)
            # 从 DataFrame 中切片出当前批次的数据
            current_batch_df = df_source.iloc[batch_start_index:batch_end_index]
            # Extract local_row_ids for this batch
            batch_local_ids = current_batch_df["local_row_id"].tolist()

            # 将批次 DataFrame 转换为字典列表 for LLM input, EXCLUDING local_row_id AND 来源 (if exists)
            llm_input_batch_list = (
                current_batch_df.drop(
                    columns=["local_row_id", source_column_name], errors="ignore"
                )
                .fillna("")
                .astype(str)
                .to_dict("records")
            )

            # 用于日志输出的起始和结束行号 (从 1 开始)
            batch_start_row_num = batch_start_index + 1
            batch_end_row_num = batch_end_index
            print(
                f"   >> 处理批次 {batch_num + 1}/{num_batches} (行 {batch_start_row_num}-{batch_end_row_num})，文件: {file_basename}"
            )

            # --- 3.3 更新详细进度 (基于批次) ---
            if update_progress_callback:
                # 进度基于已 *完成* 的批次数计算
                batches_done_in_file = batch_num + 1
                file_progress_pct = (batches_done_in_file / num_batches) * 100
                # 计算总体进度：(已完成文件数 + 当前文件内部完成度) / 总文件数
                # i 代表已完成的文件数 (0-based), i+1 是当前文件编号 (1-based)
                overall_progress_pct = int(
                    ((i + (batches_done_in_file / num_batches)) / total_files) * 100
                )
                # 限制进度最大为 99%，因为 100% 应该在保存完成后才报告
                overall_progress_pct = min(overall_progress_pct, 99)

                # 构造状态消息
                status_msg = f"处理文件 {current_file_num}/{total_files} ({file_basename}) - 完成批次 {batches_done_in_file}/{num_batches}"
                # 调用回调函数更新进度
                update_progress_callback(
                    status_msg,
                    overall_progress_pct,
                    i,
                    total_files,  # 注意：files_processed 参数仍为 i (已完成的文件数)
                )

            total_rows_attempted += len(llm_input_batch_list)  # 累加尝试处理的行数

            # --- 3.4 调用 LLM 处理批次 ---
            try:
                standardized_batch_result = extract_standardize_batch_with_llm(
                    llm_input_batch_list,
                    source_headers,
                    llm_target_columns,  # *** 修改：传入不含"来源"的列 ***
                    api_key,
                    api_endpoint,
                    max_tokens,
                    api_timeout,
                )
            except ValueError as ve:
                # 捕获由 extract_standardize_batch_with_llm 抛出的严重错误 (如上下文超限、认证失败)
                print(f"   ❌ LLM 调用期间发生严重错误 (批次 {batch_num+1}): {ve}")
                # 报告错误并停止整个任务
                if update_progress_callback:
                    update_progress_callback(
                        f"严重错误: {ve}。处理已停止。", 100, i, total_files
                    )
                raise ve  # 重新抛出异常，终止 process_files_and_consolidate
            except Exception as e:
                # 捕获 LLM 调用期间其他未预料的错误
                print(f"   ❌ LLM 调用期间发生意外错误 (批次 {batch_num+1}): {e}")
                if update_progress_callback:
                    update_progress_callback(
                        f"处理批次时发生意外错误。处理已停止。",
                        100,
                        i,
                        total_files,
                    )
                raise e  # 重新抛出异常，终止

            # --- 3.5 处理 LLM 返回结果 ---
            processed_batch_with_ids_and_source = []  # Renamed list

            if isinstance(standardized_batch_result, list):
                # 修改判断条件，允许LLM返回比输入更多的行（支持多手机号拆分）
                if len(standardized_batch_result) >= len(llm_input_batch_list):
                    # 如果LLM返回了更多行，可能是因为进行了手机号拆分
                    if len(standardized_batch_result) > len(llm_input_batch_list):
                        print(
                            f"      处理拆分的手机号数据，原始输入 {len(llm_input_batch_list)} 行，LLM返回 {len(standardized_batch_result)} 行"
                        )

                    # 建立原始行索引到可能的拆分行映射关系
                    original_row_mapping = {}
                    processed_rows = 0

                    # 遍历LLM返回的每一行结果
                    for result_idx, result_row in enumerate(standardized_batch_result):
                        if isinstance(result_row, dict):
                            # 确定此行对应的原始行索引
                            # 对于拆分行，多个结果行会对应同一个原始行
                            original_idx = min(result_idx, len(batch_local_ids) - 1)

                            # 当处理到新的原始行时，增加计数
                            if original_idx not in original_row_mapping:
                                original_row_mapping[original_idx] = 0

                            # 获取这个原始行的local_id
                            local_id = batch_local_ids[original_idx]

                            # 保存结果行信息
                            result_row["local_row_id"] = local_id
                            source_name = local_id_to_source_map.get(
                                local_id, "UNKNOWN_SOURCE"
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
                            original_idx = min(result_idx, len(batch_local_ids) - 1)
                            local_id = batch_local_ids[original_idx]

                            error_row = {
                                col: "LLM_ITEM_FORMAT_ERROR"
                                for col in llm_target_columns
                            }
                            error_row["local_row_id"] = local_id
                            error_row[source_column_name] = local_id_to_source_map.get(
                                local_id, "UNKNOWN_SOURCE"
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
                            # Exclude local_id and 来源 from error check
                            if k not in ["local_row_id", source_column_name]
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
                    print(f"      ❌ 警告: LLM 返回列表长度不匹配... 添加错误标记。")
                    for local_id in batch_local_ids:
                        error_row = {
                            col: "BATCH_LENGTH_MISMATCH" for col in llm_target_columns
                        }
                        error_row["local_row_id"] = local_id
                        error_row[source_column_name] = local_id_to_source_map.get(
                            local_id, "UNKNOWN_SOURCE"
                        )
                        consolidated_data.append(error_row)
            else:
                # Handle batch processing failure (LLM returned None)
                print(f"      ❌ 批次 {batch_num+1} LLM 处理失败... 添加错误标记。")
                for local_id in batch_local_ids:
                    error_row = {
                        col: "BATCH_PROCESSING_FAILED" for col in llm_target_columns
                    }
                    error_row["local_row_id"] = local_id
                    error_row[source_column_name] = local_id_to_source_map.get(
                        local_id, "UNKNOWN_SOURCE"
                    )
                    consolidated_data.append(error_row)

        # --- 文件处理完毕 ---
        print(
            f"   ✅ 完成处理文件 {file_basename}。大约提取了 {rows_processed_in_file_success} 条有效行。"
        )
        # (可选) 在处理完一个文件后，可以再次调用回调报告进度
        # if update_progress_callback:
        #     overall_progress_pct = int(((i + 1) / total_files) * 100)
        #     update_progress_callback(f"完成文件 {current_file_num}/{total_files}", overall_progress_pct, i + 1, total_files)

    # --- 4. 所有文件处理循环结束 ---
    # 最终进度更新 (设置为 99%，表示即将完成保存)
    # 在循环结束后不再需要报告 99%，最后的 100% 在保存成功后报告
    # if update_progress_callback:
    #     update_progress_callback(
    #         "合并结果并准备保存...", 99, total_files, total_files
    #     )

    # 打印处理总结信息
    print("\n--- 处理总结 ---")
    print(f"    提供的输入文件总数: {total_files}")
    print(f"    成功读取并处理的文件数: {files_processed_count}")
    if files_with_errors:
        print(f"    读取失败的文件: {', '.join(files_with_errors)}")
    print(f"    尝试处理的总行数: {total_rows_attempted}")
    print(f"    大约成功提取的行数: {total_rows_successfully_processed}")
    print(f"    最终输出行数: {len(consolidated_data)} (可能包含错误标记行)")

    # --- 5. 检查是否有数据需要保存 ---
    if not consolidated_data:
        print("\n❌ 未提取到任何数据。无法创建输出文件。")
        if update_progress_callback:
            update_progress_callback(
                "错误: 未提取到有效数据", 100, total_files, total_files
            )
        raise ValueError("未能成功处理任何数据。")

    # --- 6. 创建 DataFrame 并保存到 Excel ---
    # Ensure all dicts in consolidated_data have 'local_row_id' and '来源', even if errors occurred
    for row_dict in consolidated_data:
        row_dict.setdefault("local_row_id", f"missing_uuid_{uuid.uuid4()}")
        row_dict.setdefault(source_column_name, "UNKNOWN_SOURCE_FINAL")

    # *** 修改：使用包含"来源"的 final_output_columns ***
    df_final = pd.DataFrame(consolidated_data, columns=final_output_columns)

    print(f"\n💾 保存整合后的数据到: {output_file_path}")
    try:
        # 将 DataFrame 保存为 Excel 文件
        # index=False 表示不将 DataFrame 的索引写入文件
        # engine="openpyxl" 指定使用 openpyxl 引擎 (支持 .xlsx)
        # na_rep="" 将 NaN 值在 Excel 中表示为空字符串
        df_final.to_excel(output_file_path, index=False, engine="openpyxl", na_rep="")
        print(f"🎉 数据成功保存!")
        # 报告处理完成
        if update_progress_callback:
            update_progress_callback("处理完成", 100, total_files, total_files)
        # 返回输出文件的路径
        return output_file_path
    except ImportError:
        print("❌ 保存 Excel 文件需要 'openpyxl' 库。请运行: pip install openpyxl")
        raise  # 重新抛出异常
    except PermissionError:
        print(
            f"❌ 写入文件 '{output_file_path}' 时发生权限错误。请检查文件是否被其他程序打开或是否有写入权限。"
        )
        raise
    except Exception as e:
        print(f"❌ 保存 Excel 文件时发生未知错误: {e}")
        print(traceback.format_exc())
        raise


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
