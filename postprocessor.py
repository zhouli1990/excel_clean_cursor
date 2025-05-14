# -*- coding: utf-8 -*-
import pandas as pd
import requests
import json
import time
import traceback
import re
from collections import defaultdict
import openai  # 新增: 阿里云百炼 OpenAI 兼容接口
import uuid
from utils.logger import setup_logger

logger = setup_logger("postprocessor")

# Deepseek API endpoint (可以考虑也从 config 传入)
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/chat/completions"
# 阿里云百炼默认参数
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def check_duplicate_phones(
    df: pd.DataFrame, phone_col: str, remark_col: str
) -> pd.DataFrame:
    """
    检查 DataFrame 中的重复手机号，并在备注列添加标记。

    Args:
        df: 需要处理的 DataFrame。
        phone_col: 电话号码所在的列名。
        remark_col: 用于添加备注的列名。

    Returns:
        pd.DataFrame: 添加了重复手机号标记的 DataFrame。
    """
    logger.info("[Check 1] 开始检查重复手机号...")
    if phone_col not in df.columns:
        logger.warning(f"电话列 '{phone_col}' 不存在，跳过重复手机号检查。")
        return df
    if remark_col not in df.columns:
        logger.warning(f"备注列 '{remark_col}' 不存在，无法添加标记。")
        return df

    # 查找重复项，保留所有重复出现的项 (标记为 True)
    # 注意：空字符串或 None 不应被视为重复
    duplicates = df[df[phone_col].notna() & (df[phone_col] != "")].duplicated(
        subset=[phone_col], keep=False
    )

    # 获取重复手机号的索引
    duplicate_indices = duplicates[duplicates].index

    # 在备注列添加标记
    # 使用 .loc 来避免 SettingWithCopyWarning
    if not duplicate_indices.empty:
        logger.info(f"发现 {len(duplicate_indices)} 行存在重复手机号。正在添加标记...")
        # 如果备注列已有内容，则追加；否则直接写入
        existing_remarks = df.loc[duplicate_indices, remark_col].astype(str)
        new_remark = "电话号码重复"
        # 追加标记，用分号隔开 (如果已有内容)
        df.loc[duplicate_indices, remark_col] = existing_remarks.apply(
            lambda x: f"{x}; {new_remark}" if x else new_remark
        )
    else:
        logger.info("未发现重复手机号。")

    logger.info("[Check 1] 重复手机号检查完成。")
    return df


def check_duplicate_phone_and_company(
    df: pd.DataFrame, phone_col: str, company_col: str, remark_col: str
) -> pd.DataFrame:
    """
    检查 DataFrame 中手机号和公司名都重复的行，并在备注列添加标记。
    Args:
        df: 需要处理的 DataFrame。
        phone_col: 电话号码列名。
        company_col: 公司名称列名。
        remark_col: 备注列名。

    Returns:
        pd.DataFrame: 添加了标记的 DataFrame。
    """
    logger.info(f"[Check 2] 开始检查手机号+公司名重复...")
    if phone_col not in df.columns or company_col not in df.columns:
        logger.warning(
            f"警告: 电话列 '{phone_col}' 或公司列 '{company_col}' 不存在，跳过检查。"
        )
        return df
    if remark_col not in df.columns:
        logger.warning(f"备注列 '{remark_col}' 不存在，无法添加标记。")
        return df

    # 创建临时列用于比较 (忽略大小写和空格)
    temp_phone_col = "__temp_phone__"
    temp_company_col = "__temp_company__"
    df[temp_phone_col] = df[phone_col].astype(str).str.strip()
    df[temp_company_col] = df[company_col].astype(str).str.strip().str.lower()

    # 查找同时重复的项 (忽略空值)
    duplicates = df[
        df[temp_phone_col].notna()
        & (df[temp_phone_col] != "")
        & df[temp_company_col].notna()
        & (df[temp_company_col] != "")
    ].duplicated(subset=[temp_phone_col, temp_company_col], keep=False)

    duplicate_indices = duplicates[duplicates].index
    df.drop(columns=[temp_phone_col, temp_company_col], inplace=True)  # 删除临时列

    if not duplicate_indices.empty:
        logger.info(
            f"发现 {len(duplicate_indices)} 行存在手机号+公司名重复。正在添加标记..."
        )
        existing_remarks = df.loc[duplicate_indices, remark_col].astype(str)
        new_remark = "手机号+公司名重复"
        df.loc[duplicate_indices, remark_col] = existing_remarks.apply(
            lambda x: f"{x}; {new_remark}" if x else new_remark
        )
    else:
        logger.info("未发现手机号+公司名重复。")

    logger.info("[Check 2] 手机号+公司名重复检查完成。")
    return df


def check_related_companies_for_duplicate_phones_llm(
    df: pd.DataFrame,
    phone_col: str,
    company_col: str,
    remark_col: str,
    new_related_col: str,
    api_key: str = "",
    config: dict = None,
) -> pd.DataFrame:
    """
    使用百炼API检查同一手机号下不同公司名是否相关，并更新数据框。
    集成公司相似性检查和批量处理功能，不再依赖外部函数。
    """
    logger.info(f"[Check 3] 开始百炼API检查关联公司 (针对重复手机号)...")

    # 从config["llm_config"]中获取百炼API配置
    llm_config = config.get("llm_config", {})
    dashscope_api_key = llm_config.get("DASHSCOPE_API_KEY", "")
    base_url = llm_config.get("BAILIAN_BASE_URL", BAILIAN_BASE_URL)
    model_name = llm_config.get("BAILIAN_MODEL_NAME", "qwen-turbo-latest")
    batch_size = llm_config.get("BATCH_SIZE", 50)

    # 检查API密钥是否有效
    if not dashscope_api_key:
        logger.warning(
            "警告: 未找到有效的百炼API密钥(DASHSCOPE_API_KEY)，跳过关联公司检查。"
        )
        return df

    logger.info(f"使用阿里云百炼API进行关联公司检查")

    # 验证必要列存在
    if (
        phone_col not in df.columns
        or company_col not in df.columns
        or remark_col not in df.columns
    ):
        logger.warning(
            f"警告: 缺少必要的列 ('{phone_col}', '{company_col}', '{remark_col}')，跳过百炼API检查。"
        )
        return df

    # 确保新列存在
    if new_related_col not in df.columns:
        logger.info(f"创建新列: '{new_related_col}'")
        df[new_related_col] = ""
    else:
        df[new_related_col] = df[new_related_col].fillna("").astype(str)

    # 保留这些变量名以便日志输出，但不再用于筛选
    remark_phone_dup = "电话号码重复"
    remark_phone_company_dup = "手机号+公司名重复"

    # 修改筛选逻辑：直接基于重复手机号而不是依赖标记
    # 查找电话号码重复的行
    phone_duplicated = df.duplicated(subset=[phone_col], keep=False)
    candidate_df = df[
        phone_duplicated
        & df[phone_col].notna()
        & (df[phone_col] != "")
        & df[company_col].notna()
        & (df[company_col] != "")
    ].copy()  # 使用.copy()避免SettingWithCopyWarning

    if candidate_df.empty:
        logger.info(
            "没有找到需要进行百炼API关联检查的行 (没有重复手机号或相关数据不完整)。"
        )
        logger.info("[Check 3] 百炼API关联公司检查完成。")
        return df

    logger.info(
        f"找到 {len(candidate_df)} 行候选数据进行百炼API检查。开始收集公司名组..."
    )

    # 按手机号分组并收集公司名称
    grouped = candidate_df.groupby(phone_col)
    company_groups = {}

    # 收集待比较的公司名组
    for phone_number, group in grouped:
        unique_companies = group[company_col].astype(str).str.strip().unique().tolist()
        # 过滤空字符串
        unique_companies = [name for name in unique_companies if name]

        # 如果手机号下只有一个或没有不同公司名，无需比较
        if len(unique_companies) < 2:
            logger.info(f"手机号 {phone_number}: 公司名数量不足 2 个，跳过比较")
            continue

        # 将此组添加到待比较列表
        company_groups[phone_number] = unique_companies
        logger.info(f"手机号 {phone_number}: 收集 {len(unique_companies)} 个公司名")

    if not company_groups:
        logger.info("没有找到需要进行批量比较的公司名组")
        logger.info("[Check 3] 百炼API关联公司检查完成。")
        return df

    logger.info(f"使用批量处理，批处理大小: {batch_size}")

    # ----- 以下是整合的批量公司名相似性比较功能 -----

    # 将公司组分批处理
    batches = []
    current_batch = []
    for group_id, names in company_groups.items():
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.append((group_id, names))
    if current_batch:
        batches.append(current_batch)

    logger.info(f"将 {len(company_groups)} 组公司名划分为 {len(batches)} 批进行处理")

    # 存储处理结果
    results = {}

    # 新版system prompt
    system_content = (
        "你是一个企业信息去重专家。你的任务是：对同一手机号下的企业名称列表，智能判断哪些名称属于同一实际公司主体，并为每个主体只保留一个标准名称。"
        "输出唯一化后的企业名称列表（每个主体只保留一个名称），以JSON数组格式返回。"
        "例如：\n输入：['上海汉得', '汉得信息', '甄零科技']\n输出：['上海汉得', '甄零科技']\n"
        "请严格按照JSON数组格式输出结果。"
    )

    # 处理每一批
    for batch_idx, batch in enumerate(batches):
        logger.info(
            f"处理批次 {batch_idx+1}/{len(batches)}, 包含 {len(batch)} 组公司名"
        )
        batch_data = []
        for group_id, names in batch:
            unique_non_empty_names = sorted(
                list(
                    set(
                        name
                        for name in names
                        if name and isinstance(name, str) and name.strip()
                    )
                )
            )
            if len(unique_non_empty_names) >= 2:
                batch_data.append({"id": group_id, "names": unique_non_empty_names})
        if not batch_data:
            logger.info(f"批次 {batch_idx+1} 没有有效的比较组，跳过")
            continue
        # user prompt只传递企业名称列表
        for group in batch_data:
            user_content = f"请处理以下企业名称列表：{json.dumps(group['names'], ensure_ascii=False)}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            # 重试逻辑
            max_retries = 2
            retry_delay = 5
            success = False

            for attempt in range(max_retries):
                try:
                    # 初始化OpenAI客户端
                    client = openai.OpenAI(api_key=dashscope_api_key, base_url=base_url)

                    # 发送请求
                    print(
                        f"      [尝试 {attempt+1}] 发送批次 {batch_idx+1} 请求到百炼API..."
                    )
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )

                    # 提取结果
                    content_str = response.choices[0].message.content
                    print(
                        f"      [尝试 {attempt+1}] 收到批次 {batch_idx+1} 百炼API响应"
                    )

                    # 处理响应内容
                    if content_str:
                        try:
                            content_data = json.loads(content_str)
                            batch_results = []

                            # 检查返回格式是否为数组
                            if isinstance(content_data, list):
                                batch_results = content_data
                                print(
                                    f"      解析到数组格式的结果，包含 {len(batch_results)} 个项目"
                                )
                            elif (
                                isinstance(content_data, dict)
                                and "results" in content_data
                            ):
                                batch_results = content_data.get("results", [])
                                print(
                                    f"      解析到字典格式的结果，results字段包含 {len(batch_results)} 个项目"
                                )
                            else:
                                print(f"      无法识别的响应格式: {type(content_data)}")
                                print(
                                    f"      响应内容前100字符: {str(content_data)[:100]}"
                                )

                            # 处理返回的结果
                            for item in batch_results:
                                if isinstance(item, dict) and "id" in item:
                                    group_id = item.get("id")
                                    is_related = item.get("related", False)
                                    related_names = item.get("names", [])

                                    if isinstance(is_related, bool) and isinstance(
                                        related_names, list
                                    ):
                                        results[group_id] = (is_related, related_names)
                                        print(
                                            f"      组 {group_id}: 关联={is_related}, 名称={related_names}"
                                        )

                            success = True
                            print(
                                f"      批次 {batch_idx+1} 处理成功，解析了 {len(batch_results)} 组结果"
                            )
                            break

                        except json.JSONDecodeError as json_err:
                            print(f"      ❌ [批量检查] 无法解析JSON: {json_err}")
                            print(f"      响应内容前100字符: {content_str[:100]}")
                    else:
                        print("      ❌ [批量检查] 未能获取有效内容")

                except Exception as e:
                    print(
                        f"      ❌ [批量检查] 处理出错 (尝试 {attempt+1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        print(f"      ... 将在 {retry_delay} 秒后重试")
                        time.sleep(retry_delay)

            if not success:
                print(f"      ❌ 批次 {batch_idx+1} 所有重试均失败")

    print(f"      批量比较完成，获取 {len(results)} 组结果")

    # 更新DataFrame
    update_count = 0
    for phone_number, (is_related, related_names) in results.items():
        if is_related and related_names:
            # 生成逗号分隔的字符串
            related_names_str = ",".join(related_names)

            # 找到原始DataFrame中所有与当前手机号匹配，且公司名在百炼API返回的相关列表中的行的索引
            target_indices = df[
                (df[phone_col] == phone_number) & (df[company_col].isin(related_names))
            ].index

            if not target_indices.empty:
                update_count += len(target_indices)
                print(f"      更新手机号 {phone_number}: {len(target_indices)} 行")
                # 更新新列的内容
                df.loc[target_indices, new_related_col] = related_names_str

    print(f"      完成更新，共更新 {update_count} 行数据")
    print(f"   ✅ [Check 3] 百炼API关联公司检查完成。")
    return df


# --- 新增：手机号格式校验函数 ---
def validate_phone_format(
    df: pd.DataFrame, phone_col: str, remark_col: str
) -> pd.DataFrame:
    """校验指定列是否为有效的11位手机号格式，并在备注列标记错误。"""
    print(f"   >> [Check 4] 开始校验手机号格式 (列: '{phone_col}')...")
    if phone_col not in df.columns:
        print(f"   ⚠️ 警告: 电话列 '{phone_col}' 不存在，跳过格式校验。")
        return df
    if remark_col not in df.columns:
        print(f"   ⚠️ 警告: 备注列 '{remark_col}' 不存在，无法添加标记。")
        return df

    # 正则表达式：匹配以1开头的11位数字 (13*, 14*, 15*, 16*, 17*, 18*, 19*)
    phone_regex = r"^1[3-9]\d{9}$"

    # 识别无效格式 (需要是非空且不匹配正则)
    # 先将列转为字符串，并填充 NaN 为空字符串，以便应用正则
    print(df[phone_col])
    phone_series = df[phone_col].astype(str).fillna("")
    print(phone_series)
    # 使用 apply 和 re.fullmatch 检查格式
    # mask 为 True 表示格式无效 (非空且不匹配)
    invalid_mask = phone_series.apply(
        lambda x: x != "" and not bool(re.fullmatch(phone_regex, x))
    )

    invalid_indices = df.loc[invalid_mask].index

    if not invalid_indices.empty:
        print(f"      发现 {len(invalid_indices)} 行手机号格式无效。正在添加标记...")
        existing_remarks = df.loc[invalid_indices, remark_col].astype(str)
        new_remark = "手机号格式错误"
        # 注意处理备注列本身可能存在的 'nan' 字符串 (来自之前的 fillna 或数据源)
        df.loc[invalid_indices, remark_col] = existing_remarks.apply(
            lambda x: f"{x}; {new_remark}" if x and x != "nan" else new_remark
        )
    else:
        print("      未发现格式错误的手机号。")

    print("   ✅ [Check 4] 手机号格式校验完成。")
    return df


# --- 结束：手机号格式校验函数 ---


def apply_post_processing(
    df: pd.DataFrame, config: dict, id_column: str = "行ID"
) -> tuple:
    """
    重构后的后处理主流程，严格按设计方案：
    1. 全局清理
    2. 手机号格式校验
    3. 重复手机号合并
    4. LLM企业名称唯一化
    5. Sheet分类
    6. 新增Sheet有效性过滤
    7. 日志输出
    返回：处理后DataFrame、需更新的行ID集合
    """
    logger.info("--- [重构] 开始应用后处理步骤 ---")
    # 1. 全局清理
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: (
                    x.strip().replace("\n", " ").replace("\r", " ")
                    if isinstance(x, str)
                    else x
                )
            )
    logger.info("[Step 1] 全局数据清理完成")

    feishu_config = config.get("feishu_config", {})
    company_col = feishu_config.get("COMPANY_NAME_COLUMN", "企业名称")
    phone_col = feishu_config.get("PHONE_NUMBER_COLUMN", "电话")
    remark_col = feishu_config.get("REMARK_COLUMN_NAME", "备注")
    related_company_col = feishu_config.get(
        "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
    )
    record_id_col = "record_id"
    source_col = "来源"

    # 确保必要列存在
    for col in [remark_col, related_company_col, id_column]:
        if col not in df.columns:
            df[col] = ""
    if record_id_col not in df.columns:
        df[record_id_col] = ""
    if source_col not in df.columns:
        df[source_col] = ""

    # 2. 手机号格式校验
    df = validate_phone_format(df, phone_col, remark_col)
    # 标记为"手机号格式错误"的行不参与后续Sheet分类
    invalid_mask = df[remark_col].astype(str).str.contains("手机号格式错误", na=False)
    df_valid = df[~invalid_mask].copy()
    logger.info(
        f"[Step 2] 手机号格式校验后有效数据: {len(df_valid)} 行，无效: {invalid_mask.sum()} 行"
    )

    # 3. 重复手机号合并
    df_merged, _ = merge_duplicate_phones(df_valid, config, id_column)
    logger.info(f"[Step 3] 重复手机号合并后: {len(df_merged)} 行")

    # 4. LLM企业名称唯一化（直接覆盖企业名称字段）
    df_llm = llm_unique_company_names(df_merged, phone_col, company_col, config)
    logger.info(f"[Step 4] LLM企业名称唯一化后: {len(df_llm)} 行")

    # 5. Sheet分类
    # 有record_id且内容有变化进更新Sheet，无record_id进新增Sheet
    df_llm[record_id_col] = df_llm[record_id_col].fillna("").astype(str)
    df_new = df_llm[df_llm[record_id_col] == ""].copy()
    df_update = df_llm[df_llm[record_id_col] != ""].copy()
    logger.info(f"[Step 5] 新增Sheet: {len(df_new)} 行，更新Sheet: {len(df_update)} 行")

    # 6. 新增Sheet有效性过滤
    mask_company = df_new[company_col].notna() & df_new[company_col].astype(
        str
    ).str.strip().ne("")
    mask_phone = df_new[phone_col].notna() & df_new[phone_col].astype(
        str
    ).str.strip().ne("")
    df_new_filtered = df_new[mask_company | mask_phone].copy()
    logger.info(
        f"[Step 6] 新增Sheet有效性过滤后: {len(df_new_filtered)} 行，过滤掉 {len(df_new) - len(df_new_filtered)} 行"
    )

    # 7. 返回处理后DataFrame和需更新的行ID集合
    update_ids = set(df_update[id_column].tolist())
    logger.info("--- [重构] 后处理步骤完成 ---")
    return df_llm, df_new_filtered, df_update, update_ids


# 新增：LLM企业名称唯一化函数


def llm_unique_company_names(df, phone_col, company_col, config):
    """
    对每个手机号下的企业名称列表，调用LLM批处理，输出唯一化企业名称，直接覆盖企业名称字段。
    """
    # 这里只做伪实现，实际应批量调用LLM并解析结果
    # 假设LLM返回的唯一化企业名称列表为['上海汉得', '甄零科技']
    # 这里只做去重+排序模拟
    df = df.copy()
    for phone, group in df.groupby(phone_col):
        names = group[company_col].astype(str).str.strip().unique().tolist()
        # 这里应调用LLM，返回唯一化后的names
        # 假设LLM返回如下（实际应用API）
        unique_names = sorted(set(names))
        # 只保留第一个名称（模拟LLM唯一化）
        unique_name = unique_names[0] if unique_names else ""
        idxs = group.index
        df.loc[idxs, company_col] = unique_name
    return df


# --- 新增：查找列名的辅助函数 ---
def find_column_by_aliases(df, aliases):
    """
    在DataFrame中查找可能的列名

    Args:
        df: DataFrame对象
        aliases: 可能的列名列表

    Returns:
        找到的列名或None
    """
    for alias in aliases:
        if alias in df.columns:
            print(f"   >> 找到列名别名: '{alias}'")
            return alias
    return None


# --- 修改：处理手机号重复的合并函数 ---
def merge_duplicate_phones(
    df: pd.DataFrame, config: dict, id_column: str = "行ID"
) -> tuple:
    """
    处理电话号码重复的情况，实现需求文档 FR3.4 中的新增子步骤逻辑:
    返回(result_df, update_ids)，update_ids为所有因合并操作（如合并企业名称、来源）而需要更新的有record_id的行的id集合。
    """
    print("   >> [Step 5] 开始处理电话号码重复合并逻辑...")

    # 从配置中读取列名
    feishu_config = config.get("feishu_config", {})
    phone_col = feishu_config.get("PHONE_NUMBER_COLUMN", "电话")
    company_col = feishu_config.get("COMPANY_NAME_COLUMN", "企业名称")
    related_company_col = feishu_config.get(
        "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
    )
    source_col = "来源"
    record_id_col = "record_id"

    # 结果DataFrame
    result_rows = []
    update_ids = set()
    ids_processed = set()

    # 按手机号分组
    for phone, group in df.groupby(phone_col):
        if phone == "" or len(group) == 0:
            continue
        # 取所有企业名称（已唯一化）
        unique_companies = group[company_col].astype(str).str.strip().unique().tolist()
        # 来源字段只取一条（优先有record_id，否则第一条）
        source_val = ""
        feishu_rows = group[group[record_id_col] != ""]
        if not feishu_rows.empty:
            source_val = feishu_rows.iloc[0][source_col]
        else:
            source_val = group.iloc[0][source_col]
        # 只保留一条（优先有record_id，否则第一条）
        if not feishu_rows.empty:
            keep_row = feishu_rows.iloc[0].copy()
            keep_row[source_col] = source_val
            keep_row[company_col] = ";".join(unique_companies)
            result_rows.append(keep_row)
            update_ids.add(keep_row[id_column])
        else:
            keep_row = group.iloc[0].copy()
            keep_row[source_col] = source_val
            keep_row[company_col] = ";".join(unique_companies)
            result_rows.append(keep_row)
        ids_processed.update(group[id_column].tolist())
    # 处理未分组到的单独手机号
    unprocessed = df[~df[id_column].isin(ids_processed)]
    if not unprocessed.empty:
        result_rows.extend(unprocessed.to_dict("records"))
    result_df = pd.DataFrame(result_rows)
    return result_df, update_ids


# --- 新增：合并同一手机号所有企业名称的函数 ---
def merge_all_companies_for_same_phone(
    df: pd.DataFrame,
    phone_col="电话",
    company_col="企业名称",
    record_id_col="record_id",
    related_company_col="关联公司名称(LLM)",
) -> pd.DataFrame:
    """
    对相同手机号的所有记录，将所有企业名称合并到一条记录中

    处理逻辑：
    1. 相同手机号的记录中，找出飞书记录（有record_id的记录）作为基准
    2. 收集该手机号下所有企业名称（既包括关联的也包括不关联的）
    3. 将所有企业名称合并并更新到基准记录上
    4. 清空关联公司信息，确保记录能进入正确的Sheet

    Args:
        df: 输入DataFrame
        phone_col: 电话列名
        company_col: 企业名称列名
        record_id_col: 记录ID列名
        related_company_col: 关联公司列名

    Returns:
        处理后的DataFrame
    """
    print("   >> [Step 6] 开始执行企业名称全合并处理...")

    # 复制DataFrame以避免修改原始数据
    df = df.copy()

    # 清理和准备数据
    df[phone_col] = df[phone_col].fillna("").astype(str).str.strip()
    df[company_col] = df[company_col].fillna("").astype(str).str.strip()
    df[record_id_col] = df[record_id_col].fillna("").astype(str).str.strip()
    df[related_company_col] = df[related_company_col].fillna("").astype(str).str.strip()

    # 结果DataFrame
    result_rows = []

    # 按电话号码分组
    phone_groups = df.groupby(phone_col)

    merged_count = 0
    skipped_count = 0

    for phone, group in phone_groups:
        if len(group) <= 1 or phone == "":  # 非重复电话或空电话
            result_rows.extend(group.to_dict("records"))
            continue

        print(f"      处理电话号码: {phone}, 找到 {len(group)} 条记录")

        # 提取所有企业名称
        all_company_names = []

        # 处理每一行的企业名称，支持分隔符
        for _, row in group.iterrows():
            company_name = row[company_col]
            if company_name:
                # 处理可能已经包含分隔符的情况
                if ";" in company_name or "；" in company_name:
                    # 统一替换中文分号为英文分号
                    company_name = company_name.replace("；", ";")
                    names = [
                        name.strip() for name in company_name.split(";") if name.strip()
                    ]
                    all_company_names.extend(names)
                else:
                    all_company_names.append(company_name)

        # 去重并排序
        unique_companies = sorted(set(all_company_names))

        if not unique_companies:  # 没有有效的企业名称
            result_rows.extend(group.iloc[0:1].to_dict("records"))
            skipped_count += 1
            continue

        # 检查是否有飞书记录，优先使用飞书记录作为基准
        feishu_records = group[group[record_id_col] != ""]

        if not feishu_records.empty:
            # 使用第一个飞书记录作为基准
            base_row = feishu_records.iloc[0].to_dict()
            print(f"      使用飞书记录 (record_id: {base_row[record_id_col]}) 作为基准")
        else:
            # 无飞书记录，使用第一行作为基准
            base_row = group.iloc[0].to_dict()
            print(f"      无飞书记录，使用第一行作为基准")

        # 合并企业名称到基准行
        merged_company_name = ";".join(unique_companies)
        print(f"      合并{len(unique_companies)}个企业名称: {merged_company_name}")

        # 更新基准行
        base_row[company_col] = merged_company_name

        # 清空关联公司信息，确保能进入正确的Sheet
        base_row[related_company_col] = ""

        result_rows.append(base_row)
        merged_count += 1

    # 转回DataFrame
    result_df = pd.DataFrame(result_rows)

    print(
        f"   企业名称全合并完成: 处理了{merged_count}个电话分组，跳过{skipped_count}个分组"
    )
    print(f"   处理前行数: {len(df)}，处理后行数: {len(result_df)}")
    print("   ✅ [Step 6] 企业名称全合并处理完成。")

    return result_df


# --- 修改：创建多Sheet页Excel文件函数 ---
def create_multi_sheet_excel(
    df_processed: pd.DataFrame,
    output_filepath: str,
    config: dict,
    df_original: pd.DataFrame = None,  # 新增参数，接收未去重的原始数据
    id_column: str = "行ID",  # 使用行ID作为唯一标识符列名
    update_ids: set = None,  # 新增参数，指定需要更新的行ID
    df_new: pd.DataFrame = None,  # 新增参数，直接传递新增Sheet数据
    df_update: pd.DataFrame = None,  # 新增参数，直接传递更新Sheet数据
) -> None:
    print(
        f"[DEBUG] create_multi_sheet_excel 收到的 update_ids 类型: {type(update_ids)}, 长度: {len(update_ids) if update_ids is not None else 'None'}"
    )
    feishu_config = config.get("feishu_config", {})
    company_col = feishu_config.get("COMPANY_NAME_COLUMN", "企业名称")
    phone_col = feishu_config.get("PHONE_NUMBER_COLUMN", "电话")
    record_id_col = "record_id"
    source_col = "来源"
    # 1. 新增Sheet: 优先用df_new参数，否则兼容老逻辑
    if df_new is not None:
        df_new_sheet = df_new.copy()
    else:
        df_new_sheet = df_processed[df_processed[record_id_col].fillna("") == ""].copy()
    # 2. 更新Sheet: 优先用df_update参数，否则兼容老逻辑
    if df_update is not None:
        df_update_sheet = df_update.copy()
    elif update_ids is not None:
        df_processed[id_column] = df_processed[id_column].astype(str).str.strip()
        update_ids_str = set(str(i).strip() for i in update_ids)
        df_update_sheet = df_processed[
            df_processed[id_column].isin(update_ids_str)
        ].copy()
    else:
        df_update_sheet = df_processed[[]].copy()
    # 3. 原始数据Sheet
    if df_original is not None:
        df_for_original_sheet = df_original.copy()
    else:
        df_for_original_sheet = df_processed.copy()
    # 4. 写入Excel
    dfs_to_write = {
        "原始数据": df_for_original_sheet,
        "新增": df_new_sheet,
        "更新": df_update_sheet,
    }
    with pd.ExcelWriter(output_filepath, engine="openpyxl") as writer:
        for sheet_name, df_to_write in dfs_to_write.items():
            if len(df_to_write) > 0:
                df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                empty_df = pd.DataFrame(columns=df_processed.columns)
                empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
