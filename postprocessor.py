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
import logging

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
                    logger.debug(
                        f"      [尝试 {attempt+1}] 发送批次 {batch_idx+1} 请求到百炼API..."
                    )
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=8192,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )

                    # 提取结果
                    content_str = response.choices[0].message.content
                    logger.debug(
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
                                logger.debug(
                                    f"      解析到数组格式的结果，包含 {len(batch_results)} 个项目"
                                )
                            elif (
                                isinstance(content_data, dict)
                                and "results" in content_data
                            ):
                                batch_results = content_data.get("results", [])
                                logger.debug(
                                    f"      解析到字典格式的结果，results字段包含 {len(batch_results)} 个项目"
                                )
                            else:
                                logger.warning(
                                    f"      无法识别的响应格式: {type(content_data)}"
                                )
                                logger.warning(
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
                                        logger.debug(
                                            f"      组 {group_id}: 关联={is_related}, 名称={related_names}"
                                        )

                            success = True
                            logger.debug(
                                f"      批次 {batch_idx+1} 处理成功，解析了 {len(batch_results)} 组结果"
                            )
                            break

                        except json.JSONDecodeError as json_err:
                            logger.warning(
                                f"      ❌ [批量检查] 无法解析JSON: {json_err}"
                            )
                            logger.warning(
                                f"      响应内容前100字符: {content_str[:100]}"
                            )
                    else:
                        logger.warning("      ❌ [批量检查] 未能获取有效内容")

                except Exception as e:
                    logger.error(
                        f"      ❌ [批量检查] 处理出错 (尝试 {attempt+1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        logger.warning(f"      ... 将在 {retry_delay} 秒后重试")
                        time.sleep(retry_delay)

            if not success:
                logger.warning(f"      ❌ 批次 {batch_idx+1} 所有重试均失败")

    logger.debug(f"      批量比较完成，获取 {len(results)} 组结果")

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
                logger.debug(
                    f"      更新手机号 {phone_number}: {len(target_indices)} 行"
                )
                # 更新新列的内容
                df.loc[target_indices, new_related_col] = related_names_str

    logger.debug(f"      完成更新，共更新 {update_count} 行数据")
    logger.info(f"   ✅ [Check 3] 百炼API关联公司检查完成。")
    return df


# --- 新增：手机号格式校验函数 ---
def validate_phone_format(
    df: pd.DataFrame, phone_col: str, remark_col: str
) -> pd.DataFrame:
    """校验指定列是否为有效的11位手机号格式，并在备注列标记错误。"""
    logger.info(f"   >> [Check 4] 开始校验手机号格式 (列: '{phone_col}')...")
    if phone_col not in df.columns:
        logger.warning(f"   ⚠️ 警告: 电话列 '{phone_col}' 不存在，跳过格式校验。")
        return df
    if remark_col not in df.columns:
        logger.warning(f"   ⚠️ 警告: 备注列 '{remark_col}' 不存在，无法添加标记。")
        return df

    # 正则表达式：匹配以1开头的11位数字 (13*, 14*, 15*, 16*, 17*, 18*, 19*)
    phone_regex = r"^1[3-9]\d{9}$"

    # 识别无效格式 (需要是非空且不匹配正则)
    # 先将列转为字符串，并填充 NaN 为空字符串，以便应用正则
    logger.debug(df[phone_col])
    phone_series = df[phone_col].astype(str).fillna("")
    logger.debug(phone_series)
    # 使用 apply 和 re.fullmatch 检查格式
    # mask 为 True 表示格式无效 (非空且不匹配)
    invalid_mask = phone_series.apply(
        lambda x: x != "" and not bool(re.fullmatch(phone_regex, x))
    )

    invalid_indices = df.loc[invalid_mask].index

    if not invalid_indices.empty:
        logger.info(
            f"      发现 {len(invalid_indices)} 行手机号格式无效。正在添加标记..."
        )
        existing_remarks = df.loc[invalid_indices, remark_col].astype(str)
        new_remark = "手机号格式错误"
        # 注意处理备注列本身可能存在的 'nan' 字符串 (来自之前的 fillna 或数据源)
        df.loc[invalid_indices, remark_col] = existing_remarks.apply(
            lambda x: f"{x}; {new_remark}" if x and x != "nan" else new_remark
        )
    else:
        logger.info("      未发现格式错误的手机号。")

    logger.info("   ✅ [Check 4] 手机号格式校验完成。")
    return df


# --- 结束：手机号格式校验函数 ---


def extract_multi_company_phones_from_raw(df, phone_col, company_col):
    """
    遍历原始数据，统计所有手机号出现次数>1的分组，
    用正则分隔符[;；,，、/&\s]+拆分所有企业名称，去重，返回{手机号: [企业名1, 企业名2, ...]}
    """
    phone_counts = df[phone_col].fillna("").astype(str).str.strip().value_counts()
    multi_phones = phone_counts[phone_counts > 1].index.tolist()
    multi_company_dict = {}
    split_pattern = re.compile(r"[;；,，、/&\s]+")
    for phone in multi_phones:
        group = df[df[phone_col].fillna("").astype(str).str.strip() == phone]
        all_names = []
        for name in group[company_col].fillna("").astype(str):
            all_names.extend([n for n in split_pattern.split(name) if n.strip()])
        unique_names = sorted(set([n.strip() for n in all_names if n.strip()]))
        if len(unique_names) > 1:
            multi_company_dict[phone] = unique_names
    return multi_company_dict


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

    # 新增：遍历原始数据，提取需要唯一化的手机号及企业名列表
    multi_company_dict = extract_multi_company_phones_from_raw(
        df_valid, phone_col, company_col
    )
    logger.info(f"[Step 3.5] 需唯一化手机号数: {len(multi_company_dict)}")

    # 4. LLM企业名称唯一化（仅对multi_company_dict中的手机号处理）
    df_llm = llm_unique_company_names(
        df_merged, phone_col, company_col, config, multi_company_dict
    )
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


def llm_unique_company_names(
    df, phone_col, company_col, config, multi_company_dict=None
):
    """
    仅对multi_company_dict中的手机号做LLM唯一化，其他手机号直接保留。
    实际调用百炼大模型Batch接口。
    """
    import json
    import time

    df = df.copy()
    if not multi_company_dict:
        logger.info("[LLM唯一化] multi_company_dict 为空，跳过LLM唯一化处理。")
        return df

    logger.info(
        f"[LLM唯一化] 开始处理，需要唯一化的手机号数量: {len(multi_company_dict)}"
    )

    llm_config = config.get("llm_config", {})
    dashscope_api_key = llm_config.get("DASHSCOPE_API_KEY", "")
    base_url = llm_config.get("BAILIAN_BASE_URL", BAILIAN_BASE_URL)
    model_name = llm_config.get("BAILIAN_MODEL_NAME", "qwen-turbo-latest")
    batch_size = 200  # 调整批处理大小以适应LLM限制，写死默认值200，加快响应效率
    system_content = (
        "你是一个企业信息去重专家。你的任务是：对同一手机号下的企业名称列表，判断它们是否指向同一家公司、母子公司、或属于同一集团下的紧密关联公司，并为每个主体只保留一个标准名称。"
        "输出唯一化后的企业名称列表（每个主体只保留一个名称），以JSON数组格式返回。"
        "如果企业名称为干扰项、空值、重复项，请忽略他们"
        "如果企业名称存在错别字，请纠正后判断"
        "如果企业名称是英文的名称，或者是外企，也需要保留这些企业"
        "例子1：\n输入：['上海汉得', '汉得信息', '甄零科技']\n输出：['上海汉得', '甄零科技']\n"
        "例子2：\n输入：['腾讯', '深圳市腾讯计算机系统有限公司', '腾讯云', '腾讯科技']\n输出：['腾讯']\n"
        "例子3：\n输入：['阿里巴巴', '蚂蚁金服', '阿里云', '阿里']\n输出：['阿里巴巴']\n"
        "例子4：\n输入：['京东', '京东集团', '京东2023', '京东物流']\n输出：['京东']\n"
        "例子5：\n输入：['中国移动', '中国移动（北京）', '中国移动通信', '中国移动山东分公司']\n输出：['中国移动']\n"
        "例子6：\n输入：['', '顺丰', '顺丰速运', '顺丰控股']\n输出：['顺丰']\n"
        "例子6：\n输入：['ChinaPnRCo.,Ltd', 'ChinaPnRCo.,Ltd']\n输出：['ChinaPnRCo.,Ltd']\n"
        "请严格按照JSON数组格式输出结果。"
        "只输出一维JSON数组，不允许嵌套。"
        "错误示例：输出[['A', 'B']]，这是嵌套数组，格式错误。"
    )

    phones_to_process = list(multi_company_dict.keys())
    total_phones_to_process = len(phones_to_process)
    processed_count = 0

    for i in range(0, total_phones_to_process, batch_size):
        batch_phones = phones_to_process[i : i + batch_size]
        batch_data = [
            {"id": p, "names": multi_company_dict[p]}
            for p in batch_phones
            if p in multi_company_dict  # 确保手机号在字典中
        ]

        if not batch_data:
            logger.debug(
                f"[LLM唯一化][Batch {i//batch_size + 1}] 当前批次没有有效数据，跳过。"
            )
            continue

        logger.info(
            f"[LLM唯一化][Batch {i//batch_size + 1}/{ (total_phones_to_process + batch_size -1)//batch_size }] 处理手机号: {batch_phones}"
        )
        # logger.debug(
        #     f"[LLM唯一化][Batch {i//batch_size + 1}] payload: {json.dumps(batch_data, ensure_ascii=False)}"
        # ) # payload日志可能过长，需要时打开

        llm_results_for_batch = {}

        for group_idx, group_data in enumerate(batch_data):
            phone_id = group_data["id"]
            company_names_for_llm = group_data["names"]

            # 构造单个LLM请求
            user_content = f"请处理以下企业名称列表：{json.dumps(company_names_for_llm, ensure_ascii=False)}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            # logger.debug(
            #     f"[LLM唯一化][Detail] 手机号 {phone_id} (批次内序号 {group_idx+1}) messages: {json.dumps(messages, ensure_ascii=False)}"
            # ) # messages日志可能过长

            max_retries = 2
            retry_delay = 5
            success_for_phone = False
            unique_names_result = sorted(
                set(company_names_for_llm)
            )  # 默认回退到本地去重

            for attempt in range(max_retries):
                try:
                    client = openai.OpenAI(api_key=dashscope_api_key, base_url=base_url)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=8192,  # 根据实际情况调整
                        temperature=0.1,
                        # response_format={"type": "json_object"}, # 确保LLM支持此参数
                    )
                    content_str = response.choices[0].message.content
                    logger.debug(
                        f"[LLM唯一化][输出] 手机号: {phone_id}, LLM原始响应 (尝试 {attempt+1}): {content_str}"
                    )

                    if content_str:
                        try:
                            # 尝试去除markdown代码块标记
                            if content_str.startswith(
                                "```json"
                            ) and content_str.endswith("```"):
                                content_str = content_str[7:-3].strip()
                            elif content_str.startswith("```") and content_str.endswith(
                                "```"
                            ):
                                content_str = content_str[3:-3].strip()

                            content_data = json.loads(content_str)

                            # 根据LLM实际返回格式调整解析逻辑
                            if isinstance(content_data, list):
                                parsed_names = content_data
                            elif (
                                isinstance(content_data, dict)
                                and "unique_companies" in content_data
                                and isinstance(content_data["unique_companies"], list)
                            ):
                                parsed_names = content_data["unique_companies"]
                            elif (
                                isinstance(content_data, dict)
                                and "result" in content_data
                                and isinstance(content_data["result"], list)
                            ):
                                parsed_names = content_data["result"]
                            elif (
                                isinstance(content_data, dict)
                                and "names" in content_data
                                and isinstance(content_data["names"], list)
                            ):
                                parsed_names = content_data[
                                    "names"
                                ]  # 假设 'names' 键包含列表
                            else:
                                logger.warning(
                                    f"[LLM唯一化][解析] 手机号: {phone_id} 响应JSON结构未知 ({type(content_data)})，使用原始名称本地去重。响应: {content_str[:200]}"
                                )
                                parsed_names = (
                                    company_names_for_llm  # 保留原始，后续本地去重
                                )

                            unique_names_result = sorted(
                                set(
                                    [
                                        str(n).strip()
                                        for n in parsed_names
                                        if str(n).strip()
                                    ]
                                )
                            )
                            success_for_phone = True
                            logger.info(
                                f"[LLM唯一化][成功] 手机号: {phone_id}, 唯一化后名称: {unique_names_result}"
                            )
                            break  # 当前手机号处理成功，跳出重试循环
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"[LLM唯一化][JSON错误] 手机号: {phone_id} JSON解析失败 (尝试 {attempt+1}): {e}. 响应: {content_str[:200]}"
                            )
                        except Exception as e_parse:  # 其他解析阶段错误
                            logger.warning(
                                f"[LLM唯一化][解析错误] 手机号: {phone_id} 解析响应时出错 (尝试 {attempt+1}): {e_parse}. 响应: {content_str[:200]}"
                            )
                    else:
                        logger.warning(
                            f"[LLM唯一化][空响应] 手机号: {phone_id} LLM响应为空 (尝试 {attempt+1})."
                        )

                except openai.APIConnectionError as e_conn:
                    logger.error(
                        f"[LLM唯一化][连接错误] 手机号: {phone_id} (尝试 {attempt+1}): {e_conn}"
                    )
                except openai.RateLimitError as e_rate:
                    logger.error(
                        f"[LLM唯一化][速率限制] 手机号: {phone_id} (尝试 {attempt+1}): {e_rate}"
                    )
                except openai.APIStatusError as e_status:
                    logger.error(
                        f"[LLM唯一化][API状态错误] 手机号: {phone_id} (尝试 {attempt+1}): {e_status}. Status: {e_status.status_code}, Response: {e_status.response}"
                    )
                except Exception as e_gen:
                    logger.error(
                        f"[LLM唯一化][未知错误] 手机号: {phone_id} LLM接口调用失败 (尝试 {attempt+1}): {e_gen}"
                    )

                if attempt < max_retries - 1:
                    logger.info(
                        f"[LLM唯一化][重试] 手机号: {phone_id}, {retry_delay} 秒后重试..."
                    )
                    time.sleep(retry_delay)

            if not success_for_phone:
                logger.warning(
                    f"[LLM唯一化][回退] 手机号: {phone_id} 所有重试失败，使用本地去重结果: {unique_names_result}"
                )

            llm_results_for_batch[phone_id] = unique_names_result
            processed_count += 1

        # 回写DataFrame
        for phone_num, final_unique_names in llm_results_for_batch.items():
            # 定位到原始df中需要更新的行
            # 注意：df可能在之前的步骤中已经被修改，这里的df是传入此函数的副本的副本
            # 应该直接操作传入此函数的df副本
            idxs = df[df[phone_col] == phone_num].index
            if not idxs.empty:
                final_unique_name_str = (
                    ";".join(final_unique_names) if final_unique_names else ""
                )
                df.loc[idxs, company_col] = final_unique_name_str
                logger.debug(
                    f"[LLM唯一化][更新DF] 手机号: {phone_num}, 更新企业名称为: '{final_unique_name_str}' (影响 {len(idxs)} 行)"
                )
            else:
                logger.warning(
                    f"[LLM唯一化][更新DF警告] 手机号: {phone_num} 在DataFrame中未找到对应记录，无法更新。"
                )

        logger.info(
            f"[LLM唯一化] 已处理 {processed_count}/{total_phones_to_process} 个手机号。"
        )

    # 对于不在multi_company_dict中的手机号，其企业名称保持不变，无需额外处理或日志
    # 此处的循环仅用于演示之前的跳过逻辑，现在可以直接移除
    # for phone, group in df.groupby(phone_col):
    #     if phone not in multi_company_dict:
    #         # logger.debug(
    #         #     f"[LLM唯一化][跳过] 手机号: {phone}, 企业名称: {group[company_col].iloc[0]}"
    #         # )
    #         continue
    logger.info(f"[LLM唯一化] 处理完成。最终DataFrame行数: {len(df)}")
    return df


# --- 修改：处理手机号重复的合并函数 ---
def merge_duplicate_phones(
    df: pd.DataFrame, config: dict, id_column: str = "行ID"
) -> tuple:
    """
    处理电话号码重复的情况，实现需求文档 FR3.4 中的新增子步骤逻辑:
    返回(result_df, update_ids)，update_ids为所有因合并操作（如合并企业名称、来源）而需要更新的有record_id的行的id集合。
    """
    logger.info("   >> [Step 5] 开始处理电话号码重复合并逻辑...")

    # 新增：只保留企业名称和电话均非空的数据行
    df = df[
        df["企业名称"].notnull()
        & (df["企业名称"].astype(str).str.strip() != "")
        & df["电话"].notnull()
        & (df["电话"].astype(str).str.strip() != "")
    ].copy()
    logger.info(f"   已筛选有效数据，剩余 {len(df)} 行。")

    # 从配置中读取列名
    feishu_config = config.get("feishu_config", {})
    phone_col = feishu_config.get("PHONE_NUMBER_COLUMN", "电话")
    company_col = feishu_config.get("COMPANY_NAME_COLUMN", "企业名称")
    # related_company_col = feishu_config.get( # 这行被注释掉了或不完整，但此函数中未使用
    #     "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
    # )
    source_col = "来源"
    record_id_col = "record_id"

    logger.debug(
        f"   电话列: {phone_col}, 企业名称列: {company_col}, 来源列: {source_col}, Record ID列: {record_id_col}, 主ID列: {id_column}"
    )

    # 结果DataFrame
    result_rows = []
    update_ids = set()
    ids_processed = set()  # 用于记录已处理的行ID

    logger.info(f"   开始按电话号码 '{phone_col}' 分组处理数据...")
    # 按手机号分组
    for phone, group in df.groupby(phone_col):
        if phone == "" or len(group) == 0:  # 忽略空电话号码或空组
            logger.debug(f"   跳过空电话号码或空组: {phone}")
            continue

        logger.debug(f"   处理电话: {phone}, 包含 {len(group)} 条记录")

        # 取所有企业名称（已唯一化）
        unique_companies = group[company_col].astype(str).str.strip().unique().tolist()
        logger.debug(f"     唯一企业名称: {unique_companies}")

        # 来源字段只取一条（优先有record_id，否则第一条）
        source_val = ""
        feishu_rows = group[
            group[record_id_col].notna() & (group[record_id_col] != "")
        ]  # 筛选出有效的飞书记录（有 record_id 且不为空）

        if not feishu_rows.empty:
            source_val = feishu_rows.iloc[0][source_col]
            logger.debug(
                f"     从飞书记录 (ID: {feishu_rows.iloc[0][record_id_col]}) 获取来源: {source_val}"
            )
        elif not group.empty:  # 确保组不为空
            source_val = group.iloc[0][source_col]
            logger.debug(f"     无飞书记录，从组内第一条记录获取来源: {source_val}")
        else:
            logger.warning(f"     电话 {phone} 的组为空，无法获取来源值。")
            # 可以选择跳过此组或赋默认值

        # 只保留一条记录（优先有record_id，否则第一条）
        current_row_data = {}  # 用于存储当前处理行的字典数据
        if not feishu_rows.empty:
            keep_row_series = feishu_rows.iloc[0].copy()  # 复制飞书记录的第一行
            logger.debug(
                f"     保留飞书记录 (ID: {keep_row_series[record_id_col]}) 作为基础"
            )
            keep_row_series[source_col] = source_val
            keep_row_series[company_col] = ";".join(unique_companies)  # 合并企业名称
            current_row_data = keep_row_series.to_dict()  # 转换为字典
            update_ids.add(
                str(keep_row_series[id_column])
            )  # 确保ID是字符串，与ids_processed中的类型一致
            logger.debug(f"       转换为字典: {current_row_data}")
            logger.debug(f"       行ID {keep_row_series[id_column]} 加入 update_ids")
        elif not group.empty:  # 确保组不为空
            keep_row_series = group.iloc[0].copy()  # 复制组内第一行
            logger.debug(
                f"     无飞书记录，保留组内第一条记录 (ID: {keep_row_series[id_column]}) 作为基础"
            )
            keep_row_series[source_col] = source_val
            keep_row_series[company_col] = ";".join(unique_companies)  # 合并企业名称
            current_row_data = keep_row_series.to_dict()  # 转换为字典
            logger.debug(f"       转换为字典: {current_row_data}")
            # 非飞书记录（无record_id）通常不直接"更新"，但如果它们通过id_column被追踪，也记录
        else:
            logger.warning(f"     电话 {phone} 的组为空，无法选择保留行。")
            continue  # 跳过此空电话号码组的处理

        result_rows.append(current_row_data)
        # 确保添加到ids_processed的ID是字符串类型，与update_ids中的类型一致
        ids_processed.update(group[id_column].astype(str).tolist())
        logger.debug(
            f"     电话 {phone} 处理完毕, {len(group[id_column])} 个行ID加入ids_processed"
        )

    logger.info(f"   电话号码分组处理完成。共处理 {len(ids_processed)} 条记录。")

    # 处理未分组到的单独手机号 (这些是没有重复的，或者由于某种原因未被groupby处理)
    # 确保比较的ID是字符串类型
    unprocessed_df = df[~df[id_column].astype(str).isin(ids_processed)]
    if not unprocessed_df.empty:
        logger.info(
            f"   发现 {len(unprocessed_df)} 条未在分组中处理的记录，将其直接添加到结果中。"
        )
        unprocessed_records = unprocessed_df.to_dict("records")
        result_rows.extend(unprocessed_records)
        logger.debug(f"     添加的未处理记录示例 (最多3条): {unprocessed_records[:3]}")
    else:
        logger.info("   所有记录均已在分组中处理完毕。")

    logger.info(f"   准备从 {len(result_rows)} 条记录创建最终DataFrame...")
    # 现在 result_rows 中的所有元素都应该是字典
    if not result_rows:  # 如果 result_rows 为空，创建一个空的DataFrame避免错误
        logger.warning("   result_rows 为空，将返回一个空的DataFrame。")
        # 需要确定空DataFrame的列结构，可以从原始df获取，或者定义一个最小列集合
        # 为简单起见，如果df不为空，则使用df的列，否则定义基本列
        if not df.empty:
            result_df = pd.DataFrame(columns=df.columns)
        else:  # Fallback if df is also empty
            result_df = pd.DataFrame(
                columns=[phone_col, company_col, source_col, record_id_col, id_column]
            )
    else:
        try:
            result_df = pd.DataFrame(result_rows)
            logger.info(f"   最终DataFrame创建成功，包含 {len(result_df)} 行。")
        except Exception as e:
            logger.error(f"   从result_rows创建DataFrame失败: {e}")
            logger.error(
                f"   result_rows 内容 (前3条): {result_rows[:3]}"
            )  # 打印部分内容帮助调试
            # 异常发生时，也返回一个有结构的空DataFrame，或者根据策略抛出异常
            if not df.empty:
                result_df = pd.DataFrame(columns=df.columns)
            else:
                result_df = pd.DataFrame(
                    columns=[
                        phone_col,
                        company_col,
                        source_col,
                        record_id_col,
                        id_column,
                    ]
                )

    logger.info(
        f"   电话号码重复合并逻辑处理完成。返回 {len(result_df)} 行数据，{len(update_ids)} 个待更新ID。"
    )
    return result_df, update_ids


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
    task_id_from_config = config.get("task_id", "N/A")  # 尝试从config获取task_id
    task_info = f"[任务 {task_id_from_config}] " if task_id_from_config != "N/A" else ""

    logger.debug(
        f"{task_info}[DEBUG] create_multi_sheet_excel 收到的 update_ids 类型: {type(update_ids)}, 长度: {len(update_ids) if update_ids is not None else 'None'}"
    )
    feishu_config = config.get("feishu_config", {})
    # company_col = feishu_config.get("COMPANY_NAME_COLUMN", "企业名称") # 本函数内未直接使用
    # phone_col = feishu_config.get("PHONE_NUMBER_COLUMN", "电话") # 本函数内未直接使用
    record_id_col = "record_id"  # 假设 record_id 列名固定
    # source_col = "来源" # 本函数内未直接使用

    # 定义需要排除的列
    EXCLUDED_COLS_NEW_SHEET = ["record_id", "table_id", "备注", "关联公司名称(LLM)"]
    EXCLUDED_COLS_UPDATE_SHEET = ["备注", "关联公司名称(LLM)"]

    # 1. 新增Sheet: 优先用df_new参数，否则兼容老逻辑
    if df_new is not None:
        df_new_sheet = df_new.copy()
    else:
        # 确保 df_processed[record_id_col] 在fillna之前是字符串类型，避免潜在的类型问题
        df_new_sheet = df_processed[
            df_processed[record_id_col].astype(str).fillna("") == ""
        ].copy()

    # 2. 更新Sheet: 优先用df_update参数，否则兼容老逻辑
    if df_update is not None:
        df_update_sheet = df_update.copy()
    elif (
        update_ids is not None and id_column in df_processed.columns
    ):  # 增加id_column存在性检查
        # 确保 df_processed[id_column] 和 update_ids 中的元素类型一致 (字符串)
        df_processed[id_column] = df_processed[id_column].astype(str).str.strip()
        update_ids_str = set(str(i).strip() for i in update_ids)
        df_update_sheet = df_processed[
            df_processed[id_column].isin(update_ids_str)
        ].copy()
    else:
        # 如果 update_ids 为 None 或 id_column 不存在，创建一个包含所有列的空DataFrame
        df_update_sheet = pd.DataFrame(
            columns=df_processed.columns if not df_processed.empty else None
        )

    # 3. 原始数据Sheet
    if df_original is not None:
        df_for_original_sheet = df_original.copy()
    else:
        # 如果 df_original 未提供，使用 df_processed 作为原始数据的近似
        # 这在某些旧调用流程中可能是必要的
        df_for_original_sheet = df_processed.copy()

    # 4. 写入Excel
    dfs_to_write = {
        "原始数据": df_for_original_sheet,
        "新增": df_new_sheet,
        "更新": df_update_sheet,
    }

    logger.info(f"{task_info}准备写入Excel文件: {output_filepath}")
    try:
        with pd.ExcelWriter(
            output_filepath,
            engine="openpyxl",
            engine_kwargs={"write_only": True},  # 流式写入
        ) as writer:
            for sheet_name, df_current_sheet_original in dfs_to_write.items():
                if df_current_sheet_original is None:
                    logger.warning(
                        f"{task_info}Sheet '{sheet_name}' 的DataFrame为None，将写入空Sheet页。"
                    )
                    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                    continue

                # 操作副本以进行列筛选
                df_current_sheet_to_write = df_current_sheet_original.copy()

                current_columns = list(df_current_sheet_to_write.columns)
                columns_to_write = current_columns  # 默认写入所有列

                if sheet_name == "新增":
                    columns_to_write = [
                        col
                        for col in current_columns
                        if col not in EXCLUDED_COLS_NEW_SHEET
                    ]
                    logger.info(
                        f"{task_info}为 '新增' Sheet页筛选列，保留: {columns_to_write}"
                    )
                elif sheet_name == "更新":
                    columns_to_write = [
                        col
                        for col in current_columns
                        if col not in EXCLUDED_COLS_UPDATE_SHEET
                    ]
                    logger.info(
                        f"{task_info}为 '更新' Sheet页筛选列，保留: {columns_to_write}"
                    )
                elif sheet_name == "原始数据":
                    logger.info(
                        f"{task_info}为 '原始数据' Sheet页保留所有列: {columns_to_write}"
                    )
                else:
                    # 对于其他未明确指定规则的Sheet（如果将来有的话），默认也写入所有列
                    logger.info(
                        f"{task_info}为 '{sheet_name}' Sheet页保留所有列: {columns_to_write}"
                    )

                if df_current_sheet_to_write.empty:
                    logger.info(
                        f"{task_info}Sheet '{sheet_name}' 数据为空，将写入空Sheet页 (保留筛选后的列结构或原始列结构)。"
                    )
                    # 即使数据为空，也尝试保留列结构
                    final_cols_for_empty = (
                        columns_to_write if columns_to_write else current_columns
                    )
                    pd.DataFrame(columns=final_cols_for_empty).to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )
                    continue

                if not columns_to_write:
                    logger.warning(
                        f"{task_info}Sheet '{sheet_name}' (数据行数: {len(df_current_sheet_to_write)}) 筛选后没有可写入的列。将写入空Sheet页保留Sheet名。"
                    )
                    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                    continue

                # 确保 columns_to_write 中的所有列都实际存在于 df_current_sheet_to_write 中
                valid_columns_to_write = [
                    col
                    for col in columns_to_write
                    if col in df_current_sheet_to_write.columns
                ]

                if not valid_columns_to_write:
                    logger.warning(
                        f"{task_info}Sheet '{sheet_name}' (数据行数: {len(df_current_sheet_to_write)}) 的目标列 {columns_to_write} 在DataFrame中均不存在。将写入空Sheet页保留Sheet名。"
                    )
                    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                    continue

                logger.debug(
                    f"{task_info}写入Sheet '{sheet_name}'，共 {len(df_current_sheet_to_write)} 行 {len(valid_columns_to_write)} 列数据 (筛选后)"
                )
                df_current_sheet_to_write.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    columns=valid_columns_to_write,
                )
                logger.info(
                    f"{task_info}已写入Sheet '{sheet_name}'，共 {len(df_current_sheet_to_write)} 行数据 (使用筛选后列)"
                )
        logger.info(f"{task_info}成功写入所有Sheet到: {output_filepath}")
    except Exception as e:
        logger.error(
            f"{task_info}创建多Sheet页Excel文件失败: {output_filepath} - {e}",
            exc_info=True,
        )
        # 根据实际需求，这里可以决定是否抛出异常或返回状态


# --- 旧的函数，将被 utils/file_utils.py 中的版本替代或本函数将被项目统一调用 ---
# def create_multi_sheet_excel_old(...):
# pass
