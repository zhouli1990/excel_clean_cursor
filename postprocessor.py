# -*- coding: utf-8 -*-
import pandas as pd
import requests
import json
import time
import traceback
import re
from collections import defaultdict

# Deepseek API endpoint (可以考虑也从 config 传入)
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/chat/completions"


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
    print("   >> [Check 1] 开始检查重复手机号...")
    if phone_col not in df.columns:
        print(f"   ⚠️ 警告: 电话列 '{phone_col}' 不存在，跳过重复手机号检查。")
        return df
    if remark_col not in df.columns:
        print(f"   ⚠️ 警告: 备注列 '{remark_col}' 不存在，无法添加标记。")
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
        print(f"      发现 {len(duplicate_indices)} 行存在重复手机号。正在添加标记...")
        # 如果备注列已有内容，则追加；否则直接写入
        existing_remarks = df.loc[duplicate_indices, remark_col].astype(str)
        new_remark = "电话号码重复"
        # 追加标记，用分号隔开 (如果已有内容)
        df.loc[duplicate_indices, remark_col] = existing_remarks.apply(
            lambda x: f"{x}; {new_remark}" if x else new_remark
        )
    else:
        print("      未发现重复手机号。")

    print("   ✅ [Check 1] 重复手机号检查完成。")
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
    print(f"   >> [Check 2] 开始检查手机号+公司名重复...")
    if phone_col not in df.columns or company_col not in df.columns:
        print(
            f"   ⚠️ 警告: 电话列 '{phone_col}' 或公司列 '{company_col}' 不存在，跳过检查。"
        )
        return df
    if remark_col not in df.columns:
        print(f"   ⚠️ 警告: 备注列 '{remark_col}' 不存在，无法添加标记。")
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
        print(
            f"      发现 {len(duplicate_indices)} 行存在手机号+公司名重复。正在添加标记..."
        )
        existing_remarks = df.loc[duplicate_indices, remark_col].astype(str)
        new_remark = "手机号+公司名重复"
        df.loc[duplicate_indices, remark_col] = existing_remarks.apply(
            lambda x: f"{x}; {new_remark}" if x else new_remark
        )
    else:
        print("      未发现手机号+公司名重复。")

    print("   ✅ [Check 2] 手机号+公司名重复检查完成。")
    return df


def check_company_similarity_deepseek(
    company_names: list[str], api_key: str
) -> tuple[bool, list[str]]:
    """
    使用 DeepSeek API 判断公司名称列表是否指向同一或关联实体。
    (您提供的代码，稍作调整以适应这里的上下文)

    Args:
        company_names: 需要比较的公司名称列表 (应至少包含两个不同的名称)。
        api_key: DeepSeek API 密钥。

    Returns:
        一个元组 (is_related, related_names):
        - is_related (bool): 如果 API 判断这些名称相关，则为 True，否则为 False。
        - related_names (list[str]): 如果 is_related 为 True，则返回被判断为相关的公司名称列表；否则返回空列表。
        在 API 调用失败或返回意外格式时，也返回 (False, [])。
    """
    # 注意：DEEPSEEK_API_ENDPOINT 在模块顶部定义并使用
    if not api_key or not api_key.startswith("sk-"):  # 基本检查 API Key 格式
        print("         ⚠️ [LLM Check Sub] API 密钥未配置或格式无效，跳过比较。")
        return False, []

    # 确保列表中的名称是唯一的，且至少有两个不同的非空名称
    unique_non_empty_names = sorted(
        list(
            set(
                name
                for name in company_names
                if name and isinstance(name, str) and name.strip()
            )
        )
    )
    if len(unique_non_empty_names) < 2:
        # print(f"      [DeepSeek Check] 有效公司名称数量不足2个 ({unique_non_empty_names})，无需比较。")
        return False, []

    print(f"         [LLM Check Sub] 准备调用 API 比较: {unique_non_empty_names}")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    user_content = json.dumps(unique_non_empty_names, ensure_ascii=False)

    # Corrected system content string formatting
    system_content = (
        "你是一个用于判断公司名称关联性的专家。请分析以下公司名称列表，判断它们是否指向同一家公司、母子公司、或属于同一集团下的紧密关联公司。"
        "\nPlease parse the output in JSON format："
        "\nEXAMPLE INPUT:['阿里巴巴集团控股有限公司', '蚂蚁科技集团股份有限公司', '盒马（中国）有限公司']"
        "\nEXAMPLE JSON OUTPUT:{\n  \"related\": true,\n  \"names\": ['阿里巴巴集团控股有限公司', '蚂蚁科技集团股份有限公司', '盒马（中国）有限公司']\n}"
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": system_content,  # Use the corrected variable
            },
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    max_retries = 2  # 减少重试次数，避免过多等待
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(
                DEEPSEEK_API_ENDPOINT, headers=headers, json=payload, timeout=90
            )
            response.raise_for_status()
            result_json = response.json()

            content_str = None
            if "choices" in result_json and len(result_json["choices"]) > 0:
                message = result_json["choices"][0].get("message", {})
                if "content" in message:
                    content_str = message["content"]
                else:
                    print(
                        "         ❌ [LLM Check Sub] 响应缺少 'message' 或 'content' 字段。"
                    )
            else:
                print("         ❌ [LLM Check Sub] 响应缺少 'choices' 字段。")

            if content_str:
                try:
                    content_data = json.loads(content_str)
                    is_related = content_data.get("related", False)
                    related_names = content_data.get("names", [])

                    if isinstance(is_related, bool) and isinstance(related_names, list):
                        print(
                            f"         <- [LLM Check Sub] API 响应解析成功: related={is_related}, names={related_names}"
                        )
                        return is_related, related_names
                    else:
                        print(
                            f"         ❌ [LLM Check Sub] 解析后的 JSON 结构类型不符合预期。"
                        )
                        return False, []
                except json.JSONDecodeError as json_err:
                    print(
                        f"         ❌ [LLM Check Sub] 无法解析响应内容中的 JSON: {json_err}"
                    )
                    print(f"            原始 content: {content_str[:200]}...")
                    return False, []
            else:
                print("         ❌ [LLM Check Sub] 未能提取有效 content。")
                return False, []

        except requests.exceptions.Timeout:
            print(
                f"         ❌ [LLM Check Sub] 请求超时 (尝试 {attempt + 1}/{max_retries})。"
            )
            if attempt == max_retries - 1:
                return False, []
            time.sleep(retry_delay)
        except requests.exceptions.HTTPError as http_err:
            print(
                f"         ❌ [LLM Check Sub] HTTP 错误 (尝试 {attempt + 1}/{max_retries}): {http_err}"
            )
            if http_err.response is not None and http_err.response.status_code in [
                401,
                403,
                400,
                429,
            ]:
                print(
                    f"         ❌ [LLM Check Sub] 遇到 {http_err.response.status_code} 错误，停止重试。"
                )
                return False, []
            if attempt == max_retries - 1:
                return False, []
            time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            print(
                f"         ❌ [LLM Check Sub] 网络请求失败 (尝试 {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                return False, []
            time.sleep(retry_delay)
        except Exception as e:
            print(f"         ❌ [LLM Check Sub] 调用时发生未知严重错误: {e}")
            print(traceback.format_exc())
            return False, []

    print("         ❌ [LLM Check Sub] 所有重试均失败。")
    return False, []


def check_related_companies_for_duplicate_phones_llm(
    df: pd.DataFrame,
    phone_col: str,
    company_col: str,
    remark_col: str,
    new_related_col: str,  # 新增列的名称
    api_key: str,
) -> pd.DataFrame:
    """
    对于备注中标记了"电话号码重复"但未标记"手机号+公司名重复"的行，
    使用 LLM 检查同一手机号下的不同公司名是否相关，并在新列中记录相关名称。
    Args:
        df: 需要处理的 DataFrame。
        phone_col: 电话号码列名。
        company_col: 公司名称列名。
        remark_col: 备注列名。
        new_related_col: 用于存储 LLM 判断的相关公司名称的新列名。
        api_key: DeepSeek API 密钥。

    Returns:
        pd.DataFrame: 添加了关联公司名称列和标记的 DataFrame。
    """
    print(f"   >> [Check 3] 开始 LLM 检查关联公司 (针对重复手机号)...")
    if not api_key or not api_key.startswith("sk-"):
        print("   ⚠️ 警告: DeepSeek API Key 无效，跳过 LLM 关联公司检查。")
        return df
    if (
        phone_col not in df.columns
        or company_col not in df.columns
        or remark_col not in df.columns
    ):
        print(
            f"   ⚠️ 警告: 缺少必要的列 ('{phone_col}', '{company_col}', '{remark_col}')，跳过 LLM 检查。"
        )
        return df

    # 确保新列存在
    if new_related_col not in df.columns:
        print(f"      创建新列: '{new_related_col}'")
        df[new_related_col] = ""
    else:
        df[new_related_col] = df[new_related_col].fillna("").astype(str)

    remark_phone_dup = "电话号码重复"
    remark_phone_company_dup = "手机号+公司名重复"

    # 筛选出需要进行 LLM 检查的行：有电话重复标记，但没有手机号+公司名重复标记
    # 还需要确保电话号码和公司名不为空
    candidate_df = df[
        df[remark_col].astype(str).str.contains(remark_phone_dup)
        & (~df[remark_col].astype(str).str.contains(remark_phone_company_dup))
        & df[phone_col].notna()
        & (df[phone_col] != "")
        & df[company_col].notna()
        & (df[company_col] != "")
    ].copy()  # 使用 .copy() 避免 SettingWithCopyWarning

    if candidate_df.empty:
        print("      没有找到需要进行 LLM 关联检查的行 (电话重复但公司名不同)。")
        print(f"   ✅ [Check 3] LLM 关联公司检查完成。")
        return df

    print(f"      找到 {len(candidate_df)} 行候选数据进行 LLM 检查。按手机号分组...")

    # 按手机号分组
    grouped = candidate_df.groupby(phone_col)
    llm_checked_count = 0
    llm_related_found_count = 0

    for phone_number, group in grouped:
        unique_companies = group[company_col].astype(str).str.strip().unique().tolist()
        # 过滤掉空字符串
        unique_companies = [name for name in unique_companies if name]

        print(f"      处理手机号: {phone_number}，关联的公司名: {unique_companies}")

        # 如果该手机号下只有一个或没有不同的公司名，则无需比较
        if len(unique_companies) < 2:
            print(f"         公司名数量不足 2 个，跳过 LLM 比较。")
            continue

        # 调用 LLM 进行比较
        llm_checked_count += 1
        try:
            is_related, related_names_from_llm = check_company_similarity_deepseek(
                unique_companies, api_key
            )

            # 如果 LLM 判断相关
            if is_related and related_names_from_llm:
                llm_related_found_count += 1
                print(f"         LLM 判断相关: {related_names_from_llm}")
                # 生成逗号分隔的字符串
                related_names_str = ",".join(related_names_from_llm)

                # 找到原始 DataFrame 中所有与当前手机号匹配，
                # 且公司名在 LLM 返回的相关列表中的行的索引
                target_indices = df[
                    (df[phone_col] == phone_number)
                    & (df[company_col].isin(related_names_from_llm))
                ].index

                if not target_indices.empty:
                    print(
                        f"         正在为 {len(target_indices)} 行更新 '{new_related_col}' 列..."
                    )
                    # 更新新列的内容
                    df.loc[target_indices, new_related_col] = related_names_str
                    # (可选) 可以在备注列也加个标记
                    # remark_llm = "公司名可能关联(LLM)"
                    # existing_remarks = df.loc[target_indices, remark_col].astype(str)
                    # df.loc[target_indices, remark_col] = existing_remarks.apply(lambda x: f"{x}; {remark_llm}" if x else remark_llm)
                else:
                    print(
                        "         警告: LLM 返回了相关名称，但在原始数据中未找到对应行？"
                    )
            else:
                print(f"         LLM 判断不相关或返回无效。")

        except Exception as e:
            print(f"      ❌ 处理手机号 {phone_number} 的 LLM 检查时出错: {e}")
            traceback.print_exc()
            # 出错时跳过当前手机号的处理
            pass

    print(
        f"      完成 LLM 检查。共对 {llm_checked_count} 组手机号进行了检查，发现 {llm_related_found_count} 组可能相关。"
    )
    print(f"   ✅ [Check 3] LLM 关联公司检查完成。")
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
    phone_series = df[phone_col].astype(str).fillna("")
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


def apply_post_processing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    应用所有选定的后处理步骤。

    Args:
        df: 合并后的 DataFrame。
        config: 包含所有配置的字典, 包括 llm_config, feishu_config, 和 post_processing_config。
                post_processing_config 结构示例:
                {
                    "check_duplicate_phones": {"post_phone_col_for_dup_phones": "电话"},
                    "check_duplicate_companies": {
                        "post_phone_col_for_dup_comp": "联系方式",
                        "post_company_col_for_dup_comp": "客户名称"
                    },
                    "validate_phone_format": {"post_phone_col_for_regex": "电话"} # 新增
                }

    Returns:
        pd.DataFrame: 应用了后处理标记和清理的 DataFrame。
    """
    print("--- 开始应用后处理步骤 --- ")

    # --- 新增：全局数据清理 ---
    print("   >> [Step 0] 开始执行全局数据清理 (去空格、去换行)...")
    if df.empty:
        print("      DataFrame 为空，跳过清理。")
    else:
        try:
            for col in df.select_dtypes(
                include=["object"]
            ).columns:  # 只处理 object (通常是 string) 类型的列
                if col in df.columns:  # Double check column exists
                    original_type = df[col].dtype
                    print(f"      清理列: '{col}' (类型: {original_type})")
                    # 1. 去除前后空格
                    # 先确保是字符串类型再操作
                    df[col] = df[col].astype(str).str.strip()
                    # 2. 去除换行符 (和 \r)
                    df[col] = (
                        df[col]
                        .str.replace("\r", "", regex=False)
                        .str.replace("\n", "", regex=False)
                    )
                    # 3. 将清理后可能产生的、仅包含空白的单元格变回空字符串 (或者 NaN，如果需要)
                    df[col] = df[
                        col
                    ].str.strip()  # 再次 strip 以处理 replace 可能留下的空白
                    # df[col] = df[col].replace('', pd.NA) # 可选：变回 NaN

            # 4. 特殊处理record_id和local_row_id列中的"none"值
            record_id_col = "record_id"
            local_row_id_col = "local_row_id"

            if record_id_col in df.columns:
                print(f"      特殊处理 '{record_id_col}' 列中的 'none' 值")
                df.loc[df[record_id_col].str.lower() == "none", record_id_col] = ""

            if local_row_id_col in df.columns:
                print(f"      特殊处理 '{local_row_id_col}' 列中的 'none' 值")
                df.loc[df[local_row_id_col].str.lower() == "none", local_row_id_col] = (
                    ""
                )

            print("   ✅ [Step 0] 全局数据清理完成。")
        except Exception as clean_err:
            print(f"   ❌ [Step 0] 数据清理过程中发生错误: {clean_err}")
            traceback.print_exc()
            # 根据需要决定是否继续，这里选择继续，但数据可能未完全清理
            pass
    # --- 清理结束 ---

    # 获取后处理配置，默认为空字典
    post_config = config.get("post_processing_config", {})
    llm_config = config.get("llm_config", {})
    feishu_config = config.get("feishu_config", {})

    # 检查项列表 (config 的键)
    choices = list(post_config.keys())
    print(f"   将执行的检查: {choices}")

    remark_col = feishu_config.get("REMARK_COLUMN_NAME", "备注")
    api_key = llm_config.get("DEEPSEEK_API_KEY", "")
    # 定义新增的关联公司列名 (Check 3 使用)
    related_company_col = feishu_config.get(
        "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
    )

    # --- 准备 DataFrame ---
    # 确保备注列存在且为字符串
    if remark_col not in df.columns:
        print(f"   创建备注列: '{remark_col}'")
        df[remark_col] = ""
    else:
        # 清理步骤已经处理过 fillna 和 astype(str)，这里可以简化
        # df[remark_col] = df[remark_col].fillna("").astype(str)
        pass  # Assuming cleaning handled it

    # 确保关联公司列存在且为字符串 (如果需要执行 Check 3)
    # 注意：Check 3 依赖于 Check 2 的 check_duplicate_companies 键
    if "check_duplicate_companies" in choices:
        if related_company_col not in df.columns:
            print(f"   创建关联公司列: '{related_company_col}'")
            df[related_company_col] = ""
        else:
            # 清理步骤已经处理过 fillna 和 astype(str)，这里可以简化
            # df[related_company_col] = df[related_company_col].fillna("").astype(str)
            pass  # Assuming cleaning handled it

    # --- 按顺序执行选中的检查 ---

    # 1. 检查重复手机号
    check1_key = "check_duplicate_phones"
    if check1_key in choices:
        check1_params = post_config.get(check1_key, {})
        phone_col_1 = check1_params.get("post_phone_col_for_dup_phones")
        if phone_col_1:
            try:
                print(f"   执行 [Check 1] 使用电话列: '{phone_col_1}'")
                df = check_duplicate_phones(df, phone_col_1, remark_col)
            except Exception as e:
                print(f"❌ 检查重复手机号 ('{phone_col_1}') 时出错: {e}")
                traceback.print_exc()
        else:
            print(f"   ⚠️ 跳过 [Check 1] 因为未在前端选择有效的电话列。")
    else:
        print("   跳过 [Check 1] 检查重复手机号 (未勾选)。")

    # 2. 检查手机号+公司名重复 & 3. LLM 检查关联公司
    check2_key = "check_duplicate_companies"
    if check2_key in choices:
        check2_params = post_config.get(check2_key, {})
        phone_col_2 = check2_params.get("post_phone_col_for_dup_comp")
        company_col = check2_params.get("post_company_col_for_dup_comp")

        if phone_col_2 and company_col:
            # --- 执行 Check 2: 手机号+公司名重复 ---
            try:
                print(
                    f"   执行 [Check 2] 使用电话列: '{phone_col_2}', 公司列: '{company_col}'"
                )
                df = check_duplicate_phone_and_company(
                    df, phone_col_2, company_col, remark_col
                )
            except Exception as e:
                print(
                    f"❌ 检查手机号+公司名重复 ('{phone_col_2}', '{company_col}') 时出错: {e}"
                )
                traceback.print_exc()

            # --- 执行 Check 3: LLM 检查 ---
            try:
                print(
                    f"   执行 [Check 3] 使用电话列: '{phone_col_2}', 公司列: '{company_col}', 新列: '{related_company_col}'"
                )
                df = check_related_companies_for_duplicate_phones_llm(
                    df,
                    phone_col_2,
                    company_col,
                    remark_col,
                    related_company_col,
                    api_key,
                )
            except Exception as e:
                print(
                    f"❌ LLM 检查关联公司 ('{phone_col_2}', '{company_col}') 时出错: {e}"
                )
                traceback.print_exc()
        else:
            print(
                f"   ⚠️ 跳过 [Check 2] 和 [Check 3] 因为未在前端选择有效的电话列和公司列。"
            )
    else:
        print("   跳过 [Check 2] 检查手机号+公司名重复 (未勾选)。")
        print("   跳过 [Check 3] LLM 检查关联公司 (未勾选)。")

    # 4. 校验手机号格式
    check4_key = "validate_phone_format"
    if check4_key in choices:
        check4_params = post_config.get(check4_key, {})
        phone_col_3 = check4_params.get(
            "post_phone_col_for_regex"
        )  # Get the specific column name for this check
        if phone_col_3:
            try:
                print(f"   执行 [Check 4] 使用电话列: '{phone_col_3}'")
                df = validate_phone_format(df, phone_col_3, remark_col)
            except Exception as e:
                print(f"❌ 校验手机号格式 ('{phone_col_3}') 时出错: {e}")
                traceback.print_exc()
        else:
            print(f"   ⚠️ 跳过 [Check 4] 因为未在前端选择有效的电话列进行格式校验。")
    else:
        print("   跳过 [Check 4] 校验手机号格式 (未勾选)。")

    print("--- 后处理步骤完成 --- ")
    return df


# --- 新增：处理手机号重复的合并函数 ---
def merge_duplicate_phones(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    处理电话号码重复的情况：
    1. 电话相同但企业不同且无关联 - 合并企业名称(分号隔开)
    2. 电话相同且企业相同或有关联 - 保留一行(优先record_id)

    Args:
        df: 输入DataFrame
        config: 配置字典，包含列名等配置

    Returns:
        处理后的DataFrame
    """
    print("开始处理电话号码重复情况...")

    # 从配置中读取列名
    post_config = config.get("post_processing_config", {})
    feishu_config = config.get("feishu_config", {})

    # 获取电话列名
    dup_phones_config = post_config.get("check_duplicate_phones", {})
    phone_col = dup_phones_config.get("post_phone_col_for_dup_phones", "电话")

    # 获取公司列名
    dup_comp_config = post_config.get("check_duplicate_companies", {})
    company_col = dup_comp_config.get("post_company_col_for_dup_comp", "公司名称")

    # 确定关联公司列名
    related_company_col = feishu_config.get(
        "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
    )

    # 默认以record_id作为飞书记录ID列
    record_id_col = "record_id"
    local_row_id_col = "local_row_id"  # 添加local_row_id列标识

    print(
        f"使用列配置: 电话列='{phone_col}', 公司列='{company_col}', 关联公司列='{related_company_col}'"
    )

    # 验证必要列存在
    required_cols = [phone_col, company_col]
    for col in required_cols:
        if col not in df.columns:
            print(f"警告: 缺少必要列 '{col}'，无法处理电话号码重复")
            return df

    # 确保record_id列和local_row_id列存在
    if record_id_col not in df.columns:
        df[record_id_col] = ""
    if local_row_id_col not in df.columns:
        df[local_row_id_col] = ""

    # 清理数据
    df[phone_col] = df[phone_col].fillna("").astype(str).str.strip()
    df[company_col] = df[company_col].fillna("").astype(str).str.strip()
    df[record_id_col] = df[record_id_col].fillna("").astype(str).str.strip()
    df[local_row_id_col] = df[local_row_id_col].fillna("").astype(str).str.strip()

    # 将字符串"none"和"None"转换为空字符串
    df.loc[df[record_id_col].str.lower() == "none", record_id_col] = ""
    df.loc[df[local_row_id_col].str.lower() == "none", local_row_id_col] = ""

    # 按电话号码分组，过滤掉空电话号码
    valid_phones = df[df[phone_col] != ""]
    phone_groups = valid_phones.groupby(phone_col)

    # 创建结果DataFrame
    result_rows = []
    # 保留未处理的行(无电话号码)
    result_rows.extend(df[df[phone_col] == ""].to_dict("records"))

    # 处理每个电话号码组
    for phone, group in phone_groups:
        if len(group) == 1:  # 非重复电话，直接添加
            result_rows.append(group.iloc[0].to_dict())
            continue

        # 处理重复电话号码
        unique_companies = group[company_col].unique().tolist()

        # 情况1: 电话号码相同但企业名称也相同
        if len(unique_companies) == 1 or all(c == "" for c in unique_companies):
            # 判断是否有行包含record_id
            has_record_id = any(rid != "" for rid in group[record_id_col])
            if has_record_id:
                # 优先保留有record_id的行
                preferred_row = group[group[record_id_col] != ""].iloc[0].to_dict()
            else:
                # 都没有record_id，保留第一行
                preferred_row = group.iloc[0].to_dict()
            result_rows.append(preferred_row)

        # 情况2: 电话号码相同，企业名称不同，判断是否有关联
        else:
            # 检查是否存在关联公司信息
            has_related_info = (related_company_col in df.columns) and any(
                group[related_company_col].fillna("").astype(str).str.strip() != ""
            )

            if has_related_info:  # 有关联公司信息，视为关联
                # 优先保留有record_id的行
                if any(rid != "" for rid in group[record_id_col]):
                    preferred_row = group[group[record_id_col] != ""].iloc[0].to_dict()
                else:
                    preferred_row = group.iloc[0].to_dict()
                result_rows.append(preferred_row)
            else:  # 无关联，合并企业名称
                # 要合并的行
                # 优先选择有record_id的行作为基础行
                if any(rid != "" for rid in group[record_id_col]):
                    base_row = group[group[record_id_col] != ""].iloc[0].to_dict()
                else:
                    base_row = group.iloc[0].to_dict()

                # 收集所有非空企业名称
                all_companies = [c for c in unique_companies if c]
                # 合并企业名称
                if all_companies:
                    base_row[company_col] = ";".join(all_companies)

                result_rows.append(base_row)

    # 转回DataFrame
    result_df = pd.DataFrame(result_rows)

    print(f"电话号码重复处理完成，原始数据 {len(df)} 行，处理后 {len(result_df)} 行")
    return result_df


# --- 新增：创建多Sheet页Excel文件函数 ---
def create_multi_sheet_excel(
    df_processed: pd.DataFrame, output_filepath: str, config: dict
) -> None:
    """
    创建包含多个Sheet页的Excel文件：
    1. 原始数据 - 包含所有数据
    2. 新增 - 不含record_id的数据
    3. 更新 - 包含record_id的数据

    Args:
        df_processed: 经过后处理的DataFrame
        output_filepath: Excel文件保存路径
        config: 配置字典，包含列名等配置
    """
    print(f"开始创建多Sheet页Excel文件: {output_filepath}")

    # 首先处理电话号码重复情况
    df_merged = merge_duplicate_phones(df_processed, config)

    # 准备不同Sheet的数据
    record_id_col = "record_id"
    local_row_id_col = "local_row_id"  # 添加local_row_id列标识

    # 确保record_id列和local_row_id列存在
    if record_id_col not in df_merged.columns:
        df_merged[record_id_col] = ""
    if local_row_id_col not in df_merged.columns:
        df_merged[local_row_id_col] = ""

    # 清理record_id和local_row_id (将NaN、None和字符串"none"转为空字符串)
    df_merged[record_id_col] = (
        df_merged[record_id_col].fillna("").astype(str).str.strip()
    )
    df_merged[local_row_id_col] = (
        df_merged[local_row_id_col].fillna("").astype(str).str.strip()
    )

    # 将字符串"none"和"None"转换为空字符串
    df_merged.loc[df_merged[record_id_col].str.lower() == "none", record_id_col] = ""
    df_merged.loc[
        df_merged[local_row_id_col].str.lower() == "none", local_row_id_col
    ] = ""

    # 筛选新增数据(record_id为空)和更新数据(record_id非空)
    df_add = df_merged[df_merged[record_id_col] == ""].copy()
    df_update = df_merged[df_merged[record_id_col] != ""].copy()

    print(f"数据分类完成: 合并后共 {len(df_merged)} 行")
    print(f"- 新增数据: {len(df_add)} 行")
    print(f"- 更新数据: {len(df_update)} 行")

    # 创建一个ExcelWriter对象
    with pd.ExcelWriter(output_filepath, engine="openpyxl") as writer:
        # 将原始数据保存到第一个Sheet，命名为"原始数据"
        # 注意：这里使用的是原始后处理数据，未合并重复电话
        df_processed.to_excel(writer, sheet_name="原始数据", index=False)

        # 将新增和更新数据保存到对应Sheet
        df_add.to_excel(writer, sheet_name="新增", index=False)
        df_update.to_excel(writer, sheet_name="更新", index=False)

    print(f"多Sheet页Excel文件创建完成: {output_filepath}")
