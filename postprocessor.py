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

    Args:
        df: 需要处理的DataFrame
        phone_col: 电话列名
        company_col: 公司名列名
        remark_col: 备注列名
        new_related_col: 存储关联公司名的新列名
        api_key: 已废弃参数，保留为兼容性，实际API密钥从config获取
        config: 配置字典，包含百炼API参数

    Returns:
        更新后的DataFrame
    """
    logger.info(f"[Check 3] 开始百炼API检查关联公司 (针对重复手机号)...")

    # 从config["llm_config"]中获取百炼API配置
    llm_config = config.get("llm_config", {})
    dashscope_api_key = llm_config.get("DASHSCOPE_API_KEY", "")
    base_url = llm_config.get("BAILIAN_BASE_URL", BAILIAN_BASE_URL)
    model_name = llm_config.get("BAILIAN_MODEL_NAME", "qwen-turbo-latest")
    batch_size = llm_config.get("BATCH_SIZE", 50)

    # 记录API配置日志，方便调试
    logger.info(
        f"百炼API配置: 密钥前缀={dashscope_api_key[:5]+'...' if dashscope_api_key else 'None'}, 模型={model_name}, 批处理大小={batch_size}"
    )

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

    # 批量处理系统提示词
    system_content = """
你是一个专业的公司关联性分析专家。你的任务是判断输入的多组公司名称列表中，每一组里的公司是否相互关联。

【判断标准】
1. 明确关联：同一公司的不同名称、母子公司关系、同一集团下的公司
2. 明确不关联：完全不同的公司、仅行业相似但无实际关系的公司
3. 需要先处理可能包含多个公司名称的输入项，支持的分隔符包括：分号(;)、逗号(,)、空格、斜杠(/)

【输出要求】
必须输出有效的JSON数组，数组中每个元素对应一组公司，结构如下：
{
  "id": "组的唯一ID",
  "related": true或false,  // 布尔值，表示是否关联
  "names": []  // 数组，仅包含相互关联的公司名称(如果无关联则为空数组)
}

【示例】
输入:
[
  {"id": "group1", "names": ["阿里巴巴集团控股有限公司", "蚂蚁科技集团股份有限公司"]},
  {"id": "group2", "names": ["华为技术有限公司", "小米科技有限公司"]}
]

输出:
[
  {"id": "group1", "related": true, "names": ["阿里巴巴集团控股有限公司", "蚂蚁科技集团股份有限公司"]},
  {"id": "group2", "related": false, "names": []}
]

严格按照上述格式输出，确保JSON格式正确，每个组都有正确的id、related和names字段。
"""

    # 处理每一批
    for batch_idx, batch in enumerate(batches):
        logger.info(
            f"处理批次 {batch_idx+1}/{len(batches)}, 包含 {len(batch)} 组公司名"
        )

        # 构建用户请求内容
        batch_data = []
        for group_id, names in batch:
            # 确保列表中的名称是唯一的非空名称
            unique_non_empty_names = sorted(
                list(
                    set(
                        name
                        for name in names
                        if name and isinstance(name, str) and name.strip()
                    )
                )
            )
            if len(unique_non_empty_names) >= 2:  # 至少需要2个公司名
                batch_data.append({"id": group_id, "names": unique_non_empty_names})

        if not batch_data:
            logger.info(f"批次 {batch_idx+1} 没有有效的比较组，跳过")
            continue

        user_content = json.dumps(batch_data, ensure_ascii=False)
        logger.info(
            f"批次 {batch_idx+1} 包含 {len(batch_data)} 个有效组，准备百炼API请求"
        )

        # 重试逻辑
        max_retries = 2
        retry_delay = 5
        success = False

        for attempt in range(max_retries):
            try:
                # 初始化OpenAI客户端
                client = openai.OpenAI(api_key=dashscope_api_key, base_url=base_url)

                # 构建messages数组
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]

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
                print(f"      [尝试 {attempt+1}] 收到批次 {batch_idx+1} 百炼API响应")

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
                            isinstance(content_data, dict) and "results" in content_data
                        ):
                            batch_results = content_data.get("results", [])
                            print(
                                f"      解析到字典格式的结果，results字段包含 {len(batch_results)} 个项目"
                            )
                        else:
                            print(f"      无法识别的响应格式: {type(content_data)}")
                            print(f"      响应内容前100字符: {str(content_data)[:100]}")

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
    应用所有后处理步骤到DataFrame，包括:
    1. 全局数据清理 (去空格、去换行)
    2. 重复手机号检查
    3. 重复(手机号+公司名)检查
    4. LLM 重复手机号的公司名相似性检查
    5. 手机号格式校验
    6. 重复手机号的合并处理
    7. 相同手机号的企业名称全合并处理

    Returns:
        (pd.DataFrame, set): 后处理后的DataFrame和需更新的行ID集合
    """
    print("--- 开始应用后处理步骤 --- ")

    # 1. 全局清理（去空格、去换行）
    print("   >> [Step 0] 开始执行全局数据清理 (去空格、去换行)...")
    # 对DataFrame的每个字符串列去除首尾空格和换行
    for col in df.columns:
        if df[col].dtype == object:  # 只处理字符串列
            df[col] = df[col].apply(
                lambda x: (
                    x.strip().replace("\n", " ").replace("\r", " ")
                    if isinstance(x, str)
                    else x
                )
            )
    print("      全局数据清理完成")

    # 获取配置项
    feishu_config = config.get("feishu_config", {})
    company_col = feishu_config.get("COMPANY_NAME_COLUMN", "企业名称")
    phone_col = feishu_config.get("PHONE_NUMBER_COLUMN", "电话")
    remark_col = feishu_config.get("REMARK_COLUMN_NAME", "备注")
    related_company_col = feishu_config.get(
        "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
    )

    # 确保备注列存在
    if remark_col not in df.columns:
        df[remark_col] = ""
        print(f"   创建新列: '{remark_col}'")

    # 确保关联公司列存在
    if related_company_col not in df.columns:
        df[related_company_col] = ""
        print(f"   创建新列: '{related_company_col}'")

    # 确保ID列存在
    if id_column not in df.columns:
        df[id_column] = [str(uuid.uuid4()) for _ in range(len(df))]
        print(f"   创建新列: '{id_column}'")

    # 2. 检查重复手机号（手机号重复标记）
    print("   >> [Check 1] 开始检查重复手机号...")
    df = check_duplicate_phones(df, phone_col, remark_col)
    print("   ✅ [Check 1] 重复手机号检查完成。")

    # 3. 检查手机号+公司名重复
    print("   >> [Check 2] 开始检查手机号+公司名重复...")
    df = check_duplicate_phone_and_company(df, phone_col, company_col, remark_col)
    print("   ✅ [Check 2] 手机号+公司名重复检查完成。")

    # 4. 检查重复手机号的公司名相似性（使用百炼API）
    print("   >> [Check 3] 开始百炼API检查关联公司...")
    # 直接传递config而不是api_key
    df = check_related_companies_for_duplicate_phones_llm(
        df, phone_col, company_col, remark_col, related_company_col, config=config
    )
    print("   ✅ [Check 3] 百炼API关联公司检查完成。")

    # 5. 校验手机号格式
    print(f"   >> [Check 4] 开始校验手机号格式 (列: '{phone_col}')...")
    df = validate_phone_format(df, phone_col, remark_col)
    print("   ✅ [Check 4] 手机号格式校验完成。")

    # 6. 处理电话号码重复合并逻辑
    print("   >> [Step 5] 开始处理电话号码重复合并逻辑...")
    df, update_ids = merge_duplicate_phones(df, config, id_column)
    print("   ✅ [Step 5] 电话号码重复合并完成。")

    # # 7. 新增: 相同手机号的企业名称全合并处理
    # print("   >> [Step 6] 开始企业名称全合并处理...")
    # df = merge_all_companies_for_same_phone(
    #     df,
    #     phone_col=phone_col,
    #     company_col=company_col,
    #     record_id_col="record_id",
    #     related_company_col=related_company_col,
    # )
    # print("   ✅ [Step 6] 企业名称全合并处理完成。")

    print("--- 后处理步骤完成 ---")
    return df, update_ids


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

    # 软查找电话列和公司列
    phone_col_found = phone_col
    company_col_found = company_col

    # 如果默认列名不存在，尝试使用别名
    if phone_col not in df.columns:
        print(f"   >> 配置的电话列 '{phone_col}' 不存在，尝试查找别名...")
        phone_aliases = [
            "电话",
            "手机",
            "手机号",
            "联系电话",
            "联系方式",
            "电话号码",
            "phone",
            "mobile",
            "tel",
        ]
        phone_col_found = find_column_by_aliases(df, phone_aliases)

    if company_col not in df.columns:
        print(f"   >> 配置的公司列 '{company_col}' 不存在，尝试查找别名...")
        company_aliases = [
            "企业名称",
            "公司",
            "公司名称",
            "单位",
            "company",
            "corporation",
            "employer",
        ]
        company_col_found = find_column_by_aliases(df, company_aliases)

    # 记录使用的列
    phone_col = phone_col_found if phone_col_found else phone_col
    company_col = company_col_found if company_col_found else company_col

    print(
        f"      使用列配置: 电话='{phone_col}', 公司='{company_col}', 关联公司='{related_company_col}', 来源='{source_col}', RecordID='{record_id_col}', ID列='{id_column}'"
    )

    # 验证必要列存在
    if not phone_col or phone_col not in df.columns:
        print(
            f"      ⚠️ 警告: 缺少必要列 '电话'（未找到任何电话相关列名），无法处理电话号码重复合并"
        )
        return df, set()

    if not company_col or company_col not in df.columns:
        print(
            f"      ⚠️ 警告: 缺少必要列 '公司'（未找到任何公司相关列名），但仍将继续电话号码合并"
        )
        # 如果找不到公司列，创建空列
        df[company_col] = ""

    # 确保记录ID列存在
    if record_id_col not in df.columns:
        print(f"      ⚠️ 警告: 缺少必要列 '{record_id_col}'，创建空列")
        df[record_id_col] = ""

    # 确保来源列和关联列存在（即使为空）
    if source_col not in df.columns:
        df[source_col] = ""
    if related_company_col not in df.columns:
        df[related_company_col] = ""
    if id_column not in df.columns:  # 确保ID列存在
        print(f"      ⚠️ 警告: 缺少必要列 '{id_column}'，创建UUID列")
        df[id_column] = [str(uuid.uuid4()) for _ in range(len(df))]

    # 清理关键列数据类型和空值
    df[phone_col] = df[phone_col].fillna("").astype(str).str.strip()
    df[company_col] = df[company_col].fillna("").astype(str).str.strip()
    df[record_id_col] = df[record_id_col].fillna("").astype(str).str.strip()
    df[source_col] = df[source_col].fillna("").astype(str).str.strip()
    df[related_company_col] = df[related_company_col].fillna("").astype(str).str.strip()
    df[id_column] = df[id_column].fillna("").astype(str).str.strip()

    # 将字符串"none"和"None"转换为空字符串
    for col in [record_id_col, id_column, related_company_col]:
        df.loc[df[col].str.lower() == "none", col] = ""

    # 找出所有电话号码不为空的行进行分组
    valid_phones_df = df[df[phone_col] != ""].copy()
    phone_groups = valid_phones_df.groupby(phone_col)

    rows_to_keep = []  # 存储要保留的行的ID
    rows_to_modify = {}  # 存储需要修改的行 {ID: {col: new_value}}
    ids_processed = set()  # 跟踪已处理的ID
    update_ids = set()  # 新增：记录需要更新的有record_id的行ID

    # 处理每个电话号码组
    for phone, group in phone_groups:
        group_ids = set(group[id_column].tolist())
        # 如果组内所有ID都处理过了，跳过 (避免因修改操作导致重处理)
        if group_ids.issubset(ids_processed):
            continue

        if len(group) == 1:  # 非重复电话，直接保留
            keep_id = group.iloc[0][id_column]
            if keep_id not in ids_processed:
                rows_to_keep.append(keep_id)
                ids_processed.add(keep_id)
            continue

        # --- 处理重复电话号码 --- #
        unique_companies = [
            c for c in group[company_col].unique().tolist() if c
        ]  # 非空公司名
        unique_sources = [
            s for s in group[source_col].unique().tolist() if s
        ]  # 非空来源名
        has_record_id = any(rid != "" for rid in group[record_id_col])
        # 检查是否有关联信息（关联列非空）
        is_related_by_llm = any(rel != "" for rel in group[related_company_col])

        # 情况 A: 公司名相同或只有一个非空公司名，或 LLM 判断相关
        if len(unique_companies) <= 1 or is_related_by_llm:
            if has_record_id:
                # 保留一个有 record_id 的行
                keep_row = group[group[record_id_col] != ""].iloc[0]
            else:
                # 保留第一行
                keep_row = group.iloc[0]

            keep_id = keep_row[id_column]
            if keep_id not in ids_processed:
                rows_to_keep.append(keep_id)
                ids_processed.add(keep_id)
                # 将组内其他行的ID标记为已处理(相当于丢弃)
                ids_processed.update(group_ids - {keep_id})

        # 情况 B: 公司名不同且 LLM 判断不相关 (或未判断)
        else:
            # 合并公司名和来源
            merged_company_name = ";".join(sorted(unique_companies))
            merged_source = ";".join(sorted(unique_sources))

            if has_record_id:
                # 保留一个有 record_id 的行
                keep_row = group[group[record_id_col] != ""].iloc[0]
                keep_id = keep_row[id_column]
                # 新增逻辑：如果保留的行有 record_id，直接将其加入 update_ids
                if keep_row[record_id_col] != "":  # 确保 record_id 真的有效
                    print(
                        f"      ➡️ 情况B (公司不同且不相关): 记录 {keep_id} (record_id: {keep_row[record_id_col]}) 将被标记为更新 (无条件)。"
                    )
                    update_ids.add(keep_id)
            else:
                # 保留第一行
                keep_row = group.iloc[0]
                keep_id = keep_row[id_column]
                # 如果没有record_id，则不加入update_ids，因为更新是针对飞书记录的
                print(
                    f"      ➡️ 情况B (公司不同且不相关): 记录 {keep_id} 无 record_id，不标记为更新。"
                )

            if keep_id not in ids_processed:
                rows_to_keep.append(keep_id)
                modifications = {}
                # 仍然需要应用修改，但不作为是否加入update_ids的判断依据
                if keep_row[company_col] != merged_company_name:
                    modifications[company_col] = merged_company_name
                if keep_row[source_col] != merged_source:
                    modifications[source_col] = merged_source

                if modifications:  # 仅当确实有字段需要修改时，才记录修改动作
                    print(
                        f"      ➡️ 情况B: 对记录 {keep_id} 应用字段修改: {modifications}"
                    )
                    rows_to_modify[keep_id] = modifications
                else:
                    print(
                        f"      ➡️ 情况B: 对记录 {keep_id} 无需应用字段修改 (字段值已一致)。"
                    )

                ids_processed.add(keep_id)
                ids_processed.update(group_ids - {keep_id})
            else:
                # 如果选择保留的行已经被处理过（理论上不应发生，但作为保险），将组内其他行标记为已处理
                ids_processed.update(group_ids - {keep_id})

    # --- 构建最终结果 --- #
    # 1. 保留所有电话号码为空的行
    no_phone_df = df[df[phone_col] == ""].copy()

    # 2. 筛选出需要保留的行
    if rows_to_keep:
        kept_df = df[df[id_column].isin(rows_to_keep)].copy()
    else:
        kept_df = pd.DataFrame(columns=df.columns)

    # 3. 应用修改
    if rows_to_modify:
        print(f"      应用 {len(rows_to_modify)} 条合并修改...")
        for row_id, mods in rows_to_modify.items():
            idx = kept_df[kept_df[id_column] == row_id].index
            if not idx.empty:
                for col, value in mods.items():
                    kept_df.loc[idx, col] = value
                    print(f"         修改 {row_id}: 设置 {col} = {value}")
            else:
                print(f"      ⚠️ 警告: 找不到要修改的行 {id_column}={row_id}")

    # 4. 合并无电话号码的行和处理过的行
    result_df = pd.concat([no_phone_df, kept_df], ignore_index=True)

    print(
        f"      电话号码重复合并完成，原始数据 {len(df)} 行，处理后保留 {len(result_df)} 行"
    )
    print("   ✅ [Step 5] 电话号码重复合并完成。")
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
) -> None:
    """
    创建多Sheet页Excel文件，自动安全获取所有飞书相关列名配置，避免KeyError。
    只将update_ids中的行放入更新Sheet。
    """
    print(
        f"[DEBUG] create_multi_sheet_excel 收到的 update_ids 类型: {type(update_ids)}, 长度: {len(update_ids) if update_ids is not None else 'None'}"
    )
    feishu_config = config.get("feishu_config", {})
    company_col_config = feishu_config.get(
        "COMPANY_NAME_COLUMN", "企业名称"
    )  # 公司名列，默认"企业名称"
    phone_col_config = feishu_config.get(
        "PHONE_NUMBER_COLUMN", "电话"
    )  # 电话列，默认"电话"
    remark_col_config = feishu_config.get(
        "REMARK_COLUMN_NAME", "备注"
    )  # 备注列，默认"备注"
    related_company_col_config = feishu_config.get(
        "RELATED_COMPANY_COLUMN_NAME", "关联公司名称(LLM)"
    )  # 关联公司列，默认"关联公司名称(LLM)"

    print(f"开始创建多Sheet页Excel文件: {output_filepath}")

    # 准备列名和必需的系统列
    record_id_col = "record_id"
    source_col = "来源"
    system_cols = [record_id_col, source_col, id_column]  # 使用行ID替代local_row_id

    # 记录待处理数据的列名，用于日志输出
    columns_in_processed = list(df_processed.columns)
    print(f"处理数据中的列: {columns_in_processed}")

    # 软查找电话列和公司列
    phone_col = phone_col_config
    company_col = company_col_config
    remark_col = remark_col_config
    related_company_col = related_company_col_config

    # 列名软查找 - 电话列
    if phone_col not in df_processed.columns:
        print(f"警告: 配置的电话列 '{phone_col}' 不存在，尝试查找别名...")
        phone_aliases = [
            "电话",
            "手机",
            "手机号",
            "联系电话",
            "联系方式",
            "电话号码",
            "phone",
            "mobile",
            "tel",
        ]
        phone_col_found = find_column_by_aliases(df_processed, phone_aliases)
        if phone_col_found:
            phone_col = phone_col_found
            print(f"使用找到的电话列别名: '{phone_col}'")
        else:
            print(f"警告: 未找到任何电话相关列，将创建空列 '{phone_col_config}'")
            df_processed[phone_col_config] = ""
            phone_col = phone_col_config

    # 列名软查找 - 公司列
    if company_col not in df_processed.columns:
        print(f"警告: 配置的公司列 '{company_col}' 不存在，尝试查找别名...")
        company_aliases = [
            "企业名称",
            "公司",
            "公司名称",
            "单位",
            "company",
            "corporation",
            "employer",
        ]
        company_col_found = find_column_by_aliases(df_processed, company_aliases)
        if company_col_found:
            company_col = company_col_found
            print(f"使用找到的公司列别名: '{company_col}'")
        else:
            print(f"警告: 未找到任何公司相关列，将创建空列 '{company_col_config}'")
            df_processed[company_col_config] = ""
            company_col = company_col_config

    # 备注列和关联公司列检查
    if remark_col not in df_processed.columns:
        print(f"警告: 配置的备注列 '{remark_col}' 不存在，将创建空列")
        df_processed[remark_col] = ""

    if related_company_col not in df_processed.columns:
        print(f"警告: 配置的关联公司列 '{related_company_col}' 不存在，将创建空列")
        df_processed[related_company_col] = ""

    # 确保系统列存在
    for col in system_cols:
        if col not in df_processed.columns:
            print(f"警告: 系统列 '{col}' 不存在，将创建空列")
            df_processed[col] = ""

    phone_format_error_tag = "手机号格式错误"

    # 使用原始数据或处理后数据作为原始Sheet
    if df_original is not None:
        df_for_original_sheet = df_original.copy()
        print(
            f"使用提供的未去重原始数据生成'原始数据'Sheet: {len(df_for_original_sheet)}行"
        )
        # === 新增：merge后处理校验结果（如备注、LLM判断等） ===
        # 以唯一ID（id_column）为锚点，将df_processed的备注、LLM判断等字段合并到原始数据
        merge_cols = []
        for col in [remark_col, related_company_col]:
            if col in df_processed.columns and col not in df_for_original_sheet.columns:
                merge_cols.append(col)
        if merge_cols:
            print(f"[原始数据Sheet] merge后处理校验字段: {merge_cols}")
            df_for_original_sheet = df_for_original_sheet.merge(
                df_processed[[id_column] + merge_cols],
                on=id_column,
                how="left",
                suffixes=("", "_后处理"),
            )
        else:
            print("[原始数据Sheet] 无需merge后处理校验字段")
    else:
        df_for_original_sheet = df_processed.copy()
        print(
            f"未提供原始数据，使用处理后数据生成'原始数据'Sheet: {len(df_for_original_sheet)}行"
        )

    # 在处理所有数据之前先确保原始数据中的必要列也存在
    if phone_col not in df_for_original_sheet.columns:
        print(f"警告: 原始数据中缺少'{phone_col}'列，将添加空列")
        df_for_original_sheet[phone_col] = ""
    else:
        # 确保电话列为字符串类型
        df_for_original_sheet[phone_col] = (
            df_for_original_sheet[phone_col]
            .fillna("")
            .astype(str)
            .replace("\.0$", "", regex=True)
        )

    # --- 开始处理新增和更新Sheet数据 ---
    print("开始处理新增和更新Sheet数据...")

    # 1. 基础清理
    df_work = df_processed.copy()

    # 类型清理：统一处理所有的字符串列
    for col in df_work.columns:
        if df_work[col].dtype == object:  # 字符串列
            df_work[col] = df_work[col].fillna("").astype(str).str.strip()
            # 将"none"和"None"转换为空字符串
            df_work.loc[df_work[col].str.lower() == "none", col] = ""

    # 2. 改进的分类方法：按照设计方案文档要求
    print("开始根据业务规则优化数据分类...")

    # --- 步骤1: 基础分类 ---
    # 新增Sheet: 只包含record_id为空的记录
    df_new_initial = df_work[df_work[record_id_col].fillna("") == ""].copy()
    # 更新Sheet: 只包含update_ids中的有record_id的行
    if update_ids is not None:
        # 日志增强：打印 update_ids 和 df_work['行ID']
        print(f"[DEBUG] update_ids（前20个）: {list(update_ids)[:20]}")
        print(
            f"[DEBUG] df_work['行ID']（前20个）: {df_work[id_column].astype(str).str.strip().head(20).tolist()}"
        )
        # 检查每个 update_id 是否在 df_work['行ID'] 中
        for test_id in list(update_ids)[:20]:
            exists = any(
                df_work[id_column].astype(str).str.strip() == str(test_id).strip()
            )
            print(f"[DEBUG] update_id {test_id} 是否在df_work中: {exists}")
        # 强制类型和格式一致性
        df_work[id_column] = df_work[id_column].astype(str).str.strip()
        update_ids_str = set(str(i).strip() for i in update_ids)
        df_update = df_work[df_work[id_column].isin(update_ids_str)].copy()
        print(f"[DEBUG] df_update 长度: {len(df_update)}")
        print(f"[DEBUG] df_update 前5行: {df_update.head(5).to_dict(orient='records')}")
    else:
        df_update = df_work[[]].copy()  # 空DataFrame

    # 有record_id且有关联公司的记录 - 这些记录不进入任何sheet
    excluded_records = df_work[
        (df_work[record_id_col].fillna("") != "")
        & (df_work[related_company_col].fillna("") != "")
    ].copy()

    if not excluded_records.empty:
        print(
            f"发现{len(excluded_records)}条有record_id且有关联公司的记录，这些记录将被排除在更新Sheet之外"
        )

    print(
        f"初步分类: 新增候选{len(df_new_initial)}条, 更新{len(df_update)}条, 排除{len(excluded_records)}条"
    )

    # --- 步骤2: 数据有效性过滤 ---
    # 对新增数据应用有效性过滤（企业名称或电话至少有一个不为空）
    original_new_count = len(df_new_initial)

    # 确保企业名称列和电话列存在
    if company_col in df_new_initial.columns and phone_col in df_new_initial.columns:
        # 构建过滤条件：企业名称或电话至少有一个非空
        mask_company = df_new_initial[company_col].notna() & df_new_initial[
            company_col
        ].astype(str).str.strip().ne("")
        mask_phone = df_new_initial[phone_col].notna() & df_new_initial[
            phone_col
        ].astype(str).str.strip().ne("")

        # 应用过滤条件
        df_new = df_new_initial[mask_company | mask_phone].copy()

        # 计算被过滤掉的行数
        filtered_count = original_new_count - len(df_new)
        print(
            f"数据有效性过滤: 从'新增'Sheet过滤掉 {filtered_count} 条无效数据（企业名称和电话均为空）"
        )

        if filtered_count > 0:
            print(
                f"数据有效性过滤完成: 原始'新增'数据 {original_new_count} 条，有效数据 {len(df_new)} 条"
            )
    else:
        print(f"警告: 企业名称列或电话列缺失，跳过有效性过滤")
        df_new = df_new_initial.copy()

    print(f"最终分类: 新增{len(df_new)}条, 更新{len(df_update)}条")

    # 4. 新增：处理大数据量分表逻辑
    FEISHU_ROW_LIMIT = 50000  # 飞书表格单表最大行数限制

    # 检查新增数据是否超过单表限制
    new_data_total = len(df_new)
    if new_data_total > FEISHU_ROW_LIMIT:
        print(
            f"检测到新增数据({new_data_total}行)超过飞书单表限制({FEISHU_ROW_LIMIT}行)，将进行分表处理"
        )

        # 计算需要的分表数量
        num_tables_needed = (new_data_total + FEISHU_ROW_LIMIT - 1) // FEISHU_ROW_LIMIT
        print(f"根据数据量，计划分成{num_tables_needed}个表")

        # 准备分表DataFrames字典
        new_data_sheets = {}
        new_data_sheets["新增"] = df_new.iloc[
            :FEISHU_ROW_LIMIT
        ]  # 第一个表仍使用"新增"名称

        # 创建额外的分表
        for i in range(1, num_tables_needed):
            start_idx = i * FEISHU_ROW_LIMIT
            end_idx = min((i + 1) * FEISHU_ROW_LIMIT, new_data_total)
            sheet_name = f"新增-{i+1}"
            new_data_sheets[sheet_name] = df_new.iloc[start_idx:end_idx]
            print(
                f"创建分表 '{sheet_name}'，包含 {len(new_data_sheets[sheet_name])} 行数据"
            )

        # 更新dfs_to_write字典中的新增数据部分
        dfs_to_write = {
            "原始数据": df_for_original_sheet,
            "更新": df_update,
        }
        # 添加所有分表
        for sheet_name, df_sheet in new_data_sheets.items():
            dfs_to_write[sheet_name] = df_sheet

        print(f"数据分类完成 (使用分表):")
        print(f"- 原始数据: {len(df_for_original_sheet)} 行")
        print(f"- 更新数据: {len(df_update)} 行")
        for sheet_name, df_sheet in new_data_sheets.items():
            print(f"- {sheet_name}: {len(df_sheet)} 行")
    else:
        print(f"数据分类完成:")
        print(f"- 原始数据: {len(df_for_original_sheet)} 行")
        print(f"- 新增数据: {len(df_new)} 行")
        print(f"- 更新数据: {len(df_update)} 行")

        # 标准数据量情况下的写入字典
        dfs_to_write = {
            "原始数据": df_for_original_sheet,
            "新增": df_new,
            "更新": df_update,
        }

    # 5. 保存到Excel
    try:
        with pd.ExcelWriter(output_filepath, engine="openpyxl") as writer:
            for sheet_name, df_to_write in dfs_to_write.items():
                # --- 新增：写入前校验"来源"字段 ---
                if "来源" in df_to_write.columns:
                    missing_source = (df_to_write["来源"] == "") | (
                        df_to_write["来源"].isna()
                    )
                    if missing_source.any():
                        print(
                            f"警告: '{sheet_name}'Sheet中'来源'字段缺失行数: {missing_source.sum()}"
                        )
                        df_to_write.loc[missing_source, "来源"] = "未知来源"
                else:
                    print(
                        f"警告: '{sheet_name}'Sheet中无'来源'字段，自动创建并补全为'未知来源'"
                    )
                    df_to_write["来源"] = "未知来源"
                print(
                    f"'{sheet_name}'Sheet '来源'字段唯一值: {df_to_write['来源'].unique().tolist()}"
                )
                # --- 原有写入逻辑 ---
                if len(df_to_write) > 0:  # 只写入非空DataFrame
                    print(
                        f"准备写入{sheet_name} Sheet，包含列: {list(df_to_write.columns)}"
                    )

                    # 检查是否需要添加必要列
                    all_required_cols = system_cols + [
                        phone_col,
                        company_col,
                        remark_col,
                        related_company_col,
                    ]
                    missing_cols = [
                        col
                        for col in all_required_cols
                        if col not in df_to_write.columns
                    ]

                    if missing_cols:
                        print(
                            f"警告: '{sheet_name}'Sheet中缺少以下列: {missing_cols}，将添加空列"
                        )
                        for col in missing_cols:
                            df_to_write[col] = ""

                    # 确保电话列是文本格式
                    # 注意：这里使用的是实际发现的phone_col，而不是配置中的值
                    if phone_col in df_to_write.columns:
                        df_to_write[phone_col] = (
                            df_to_write[phone_col]
                            .fillna("")
                            .astype(str)
                            .replace("\.0$", "", regex=True)
                        )

                    # 写入Excel
                    df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # 对于空DataFrame，创建一个包含所有必要列的空Sheet
                    all_cols = list(df_processed.columns)
                    empty_df = pd.DataFrame(columns=all_cols)
                    empty_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✅ 多Sheet页Excel文件创建完成: {output_filepath}")
    except Exception as e:
        print(f"❌ 创建Excel文件时出错: {str(e)}")
        # 尝试记录更详细的错误信息
        import traceback

        print(f"错误详情: {traceback.format_exc()}")

        # 尝试创建简化版Excel文件作为备份
        try:
            simple_path = output_filepath.replace(".xlsx", "_simple.xlsx")
            print(f"尝试创建简化版Excel文件: {simple_path}")
            df_processed.to_excel(simple_path, index=False)
            print(f"✅ 已创建简化版Excel文件: {simple_path}")
        except Exception as simple_err:
            print(f"❌ 创建简化版Excel也失败: {str(simple_err)}")

        raise
