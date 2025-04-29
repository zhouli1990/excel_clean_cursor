# -*- coding: utf-8 -*-
import pandas as pd
import requests
import json
import time
import traceback
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

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": '你是一个用于判断公司名称关联性的专家。请分析以下公司名称列表，判断它们是否指向同一家公司、母子公司、或属于同一集团下的紧密关联公司。\nPlease parse the output in JSON format：\nEXAMPLE INPUT:["阿里巴巴集团控股有限公司", "蚂蚁科技集团股份有限公司", "盒马（中国）有限公司"]\nEXAMPLE JSON OUTPUT:{\n  "related": true,\n  "names": ["阿里巴巴集团控股有限公司", "蚂蚁科技集团股份有限公司", "盒马（中国）有限公司"]\n}',
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
                    }
                }

    Returns:
        pd.DataFrame: 应用了后处理标记的 DataFrame。
    """
    print("--- 开始应用后处理步骤 --- ")
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
        df[remark_col] = df[remark_col].fillna("").astype(str)

    # 确保关联公司列存在且为字符串 (如果需要执行 Check 3)
    # 注意：Check 3 依赖于 Check 2 的 check_duplicate_companies 键
    if "check_duplicate_companies" in choices:
        if related_company_col not in df.columns:
            print(f"   创建关联公司列: '{related_company_col}'")
            df[related_company_col] = ""
        else:
            df[related_company_col] = df[related_company_col].fillna("").astype(str)

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

    print("--- 后处理步骤完成 --- ")
    return df
