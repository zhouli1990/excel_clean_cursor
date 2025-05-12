from typing import Optional, List, Dict, Any, Union

# -*- coding: utf-8 -*-
import requests
import time
import pandas as pd
import uuid

# 这些配置现在由调用者传入
# APP_ID = 'cli_a36634dc16b8d00e'
# APP_SECRET = 'RoXYTnSBGGsLLyvONbSCYe15Jm6bv5Xn'
# APP_TOKEN = 'XyUFbxc8JaDkTJscEigcbkxgnqe'
# TABLE_IDS = [...]
# COMPANY_NAME_COLUMN = '企业名称'
# PHONE_NUMBER_COLUMN = '电话'
# REMARK_COLUMN_NAME = '备注'


def get_access_token(app_id, app_secret):
    """获取租户访问令牌 (Tenant Access Token)"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
    headers = {"Content-Type": "application/json"}
    payload = {"app_id": app_id, "app_secret": app_secret}
    print("   > 正在获取飞书访问令牌...")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)  # 添加超时
        resp.raise_for_status()  # 检查 HTTP 错误
        data = resp.json()
        if data.get("code") == 0:
            token = data.get("tenant_access_token")
            print("   ✅ 飞书访问令牌获取成功!")
            return token
        else:
            error_msg = data.get("msg", "未知错误")
            print(
                f"   ❌ 获取飞书访问令牌失败: Code={data.get('code')}, Msg={error_msg}"
            )
            raise Exception(f"获取飞书访问令牌失败: {error_msg}")
    except requests.exceptions.Timeout:
        print("   ❌ 请求飞书访问令牌超时。")
        raise Exception("请求飞书访问令牌超时")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ 请求飞书访问令牌时发生网络错误: {e}")
        raise Exception(f"请求飞书访问令牌网络错误: {e}")
    except Exception as e:
        print(f"   ❌ 获取飞书访问令牌时发生未知错误: {e}")
        raise Exception(f"获取飞书访问令牌未知错误: {e}")


def fetch_all_records_from_table(access_token, app_token, table_id):
    """获取单个表格的所有记录 (包含 record_id 和 table_id)"""
    all_records_data = []
    page_token = None
    BASE_URL = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
    headers = {"Authorization": f"Bearer {access_token}"}
    PAGE_SIZE = 500  # 飞书 API 单次最大限制

    print(f"      > 开始从表格 {table_id} 获取记录...")
    page_count = 0
    while True:
        page_count += 1
        params = {"page_size": PAGE_SIZE}
        if page_token:
            params["page_token"] = page_token

        print(
            f"         请求第 {page_count} 页... (Token: {'...' + page_token[-6:] if page_token else 'N/A'})"
        )
        try:
            response = requests.get(
                BASE_URL, headers=headers, params=params, timeout=60
            )  # 增加超时
            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0:
                items = data.get("data", {}).get("items", [])
                if not items and page_token is None and not all_records_data:
                    print(f"      ⚠️ 表格 {table_id} 为空或首次请求无数据返回。")

                # 提取字段和 record_id
                for item in items:
                    record_id = item.get("record_id")
                    fields = item.get("fields", {})
                    if record_id:
                        # 将 record_id 添加到字段字典中，方便后续处理
                        fields["record_id"] = record_id
                        # 新增: 将 table_id 添加到字段字典中
                        fields["table_id"] = table_id
                        all_records_data.append(fields)
                    else:
                        print(f"      ⚠️ 发现缺少 record_id 的记录: {item}")

                has_more = data.get("data", {}).get("has_more", False)
                page_token = data.get("data", {}).get("page_token")

                print(
                    f"         本页获取 {len(items)} 条记录。累计: {len(all_records_data)} 条。HasMore={has_more}"
                )

                if has_more:
                    # 飞书限制频率较高，适当增加延时避免触发流控 (e.g., 200ms-500ms)
                    time.sleep(0.3)
                else:
                    print(
                        f"      ✅ 表格 {table_id} 所有记录获取完毕，共 {len(all_records_data)} 条。"
                    )
                    break  # 没有更多数据了，退出循环
            else:
                # API 返回错误码
                error_code = data.get("code")
                error_msg = data.get("msg", "未知错误")
                print(
                    f"      ❌ 请求表格 {table_id} 数据失败: Code={error_code}, Msg={error_msg}"
                )
                # 特定错误处理，例如token失效
                if error_code in [99991663, 99991664, 10012]:  # 令牌无效/过期/无权限
                    print(f"      ❌ 访问令牌失效或无权限访问表格 {table_id}。")
                    raise Exception(f"访问令牌失效或无权限 ({error_code})")
                else:
                    print(f"      ❌ 遇到非致命错误，停止获取表格 {table_id}。")
                    break  # 其他错误则停止当前表格的获取

        except requests.exceptions.Timeout:
            print(
                f"      ❌ 请求表格 {table_id} 时超时 (第 {page_count} 页)，可尝试增加超时时间或检查网络。停止获取此表。"
            )
            break  # 超时，停止当前表格获取 (也可以选择重试)
        except requests.exceptions.HTTPError as http_err:
            print(
                f"      ❌ 请求表格 {table_id} 时发生 HTTP 错误 (第 {page_count} 页): {http_err}"
            )
            if http_err.response is not None and http_err.response.status_code == 403:
                print(
                    f"      ❌ 403 Forbidden - 请检查 App Token 和 Table ID 是否正确，以及应用是否有读取权限。"
                )
            break  # HTTP 错误，停止当前表格获取
        except requests.exceptions.RequestException as e:
            print(
                f"      ❌ 请求表格 {table_id} 时发生网络错误 (第 {page_count} 页): {e}"
            )
            break  # 网络错误，停止当前表格获取
        except Exception as e:
            import traceback

            print(
                f"      ❌ 处理表格 {table_id} 数据时发生未知错误 (第 {page_count} 页): {e}"
            )
            print(traceback.format_exc())
            break  # 未知错误，停止当前表格获取

    return all_records_data


def fetch_and_prepare_feishu_data(feishu_config, target_columns=None):
    """
    获取所有指定飞书表格的数据，并将其合并、准备成 DataFrame。
    如果提供了target_columns参数，将只保留record_id、table_id和这些目标列，过滤掉其他飞书特有列。

    Args:
        feishu_config (dict): 包含飞书 API 配置的字典。
        target_columns (list, optional): 需要保留的目标业务列列表。默认为None，表示保留所有列。

    Returns:
        pd.DataFrame: 包含所有表格数据的合并 DataFrame，如果出错则返回空 DataFrame。
                      DataFrame 包含 record_id 列。
    """
    print("--- 开始获取飞书数据 --- ")
    all_feishu_records = []
    try:
        # 1. 获取 Access Token
        access_token = get_access_token(
            feishu_config["APP_ID"], feishu_config["APP_SECRET"]
        )
        if not access_token:
            return pd.DataFrame()  # 获取 token 失败

        # 2. 遍历 Table IDs 获取数据
        app_token = feishu_config["APP_TOKEN"]
        table_ids = feishu_config.get("TABLE_IDS", [])
        if not table_ids:
            print("   ⚠️ 未配置飞书 Table IDs，跳过飞书数据获取。")
            return pd.DataFrame()

        print(f"   配置的 Table IDs: {table_ids}")

        for table_id in table_ids:
            print(f"   处理 Table ID: {table_id}")
            try:
                table_records = fetch_all_records_from_table(
                    access_token, app_token, table_id
                )
                if table_records:  # 只有当成功获取到数据时才添加
                    all_feishu_records.extend(table_records)
                print(
                    f"   表格 {table_id} 处理完毕。当前总记录数: {len(all_feishu_records)}"
                )
            except Exception as table_e:
                # fetch_all_records_from_table 内部已打印错误，这里决定是否继续处理下一个表
                print(
                    f"   获取表格 {table_id} 数据时遇到错误: {table_e}。将尝试继续处理下一个表格。"
                )
                # 根据需要，这里可以决定是否要完全停止 (raise table_e)
                pass

        # 3. 将所有记录转换为 DataFrame
        if not all_feishu_records:
            print("   未从任何飞书表格成功获取到数据。")
            return pd.DataFrame()
        else:
            print(
                f"   成功从飞书获取总计 {len(all_feishu_records)} 条记录。正在转换为 DataFrame..."
            )
            df_feishu = pd.DataFrame(all_feishu_records)
            print("   ✅ 飞书数据已成功转换为 DataFrame。")

            # 新增：为所有缺失或无效的行ID补充唯一UUID
            id_column = "行ID"
            if id_column in df_feishu.columns:
                empty_or_none_id_mask = (
                    df_feishu[id_column].fillna("").astype(str).str.strip() == ""
                ) | (df_feishu[id_column].astype(str).str.lower() == "none")
                num_to_fill = empty_or_none_id_mask.sum()
                if num_to_fill > 0:
                    print(
                        f"   检测到 {num_to_fill} 个空的或无效的 {id_column}，为其生成UUID..."
                    )
                    df_feishu.loc[empty_or_none_id_mask, id_column] = [
                        str(uuid.uuid4()) for _ in range(num_to_fill)
                    ]
            else:
                df_feishu[id_column] = [
                    str(uuid.uuid4()) for _ in range(len(df_feishu))
                ]
                print(f"   创建新列并填充UUID: {id_column}")

            # 4. 新增：根据target_columns过滤列
            if target_columns and isinstance(target_columns, list):
                # 确保始终保留系统必要列（修复：加入'行ID'）
                required_cols = ["record_id", "table_id", "行ID"]
                cols_to_keep = required_cols + [
                    col for col in target_columns if col in df_feishu.columns
                ]

                # 记录过滤前后的列数量
                original_cols = list(df_feishu.columns)
                original_col_count = len(original_cols)

                # 应用过滤
                available_cols = [
                    col for col in cols_to_keep if col in df_feishu.columns
                ]
                if available_cols:
                    df_feishu = df_feishu[available_cols]
                    filtered_col_count = len(df_feishu.columns)
                    removed_cols = set(original_cols) - set(available_cols)

                    print(
                        f"   🔍 列过滤: 原始列数 {original_col_count} -> 过滤后列数 {filtered_col_count}"
                    )
                    print(f"   🔍 保留的列: {list(df_feishu.columns)}")
                    print(f"   🔍 过滤掉的列: {list(removed_cols)}")
                else:
                    print("   ⚠️ 过滤后没有保留任何列，返回原始DataFrame")
            else:
                print("   ℹ️ 未提供目标列表，返回所有列")

            return df_feishu

    except Exception as e:
        print(f"❌ 获取和准备飞书数据过程中发生顶层错误: {e}")
        import traceback

        print(traceback.format_exc())
        return pd.DataFrame()  # 返回空 DataFrame 表示失败


# === 新增：获取表格元数据 (包括记录数) ===


def get_table_record_count(
    access_token: str, app_token: str, table_id: str
) -> Optional[int]:
    """
    获取指定飞书多维表格的记录总数。
    使用 /records/search?page_size=1 端点获取包含 total 字段的响应。

    Args:
        access_token (str): 飞书访问令牌。
        app_token (str): Base App Token。
        table_id (str): 要查询的 Table ID。

    Returns:
        Optional[int]: 表格中的记录总数。如果获取失败，则返回 None。
    """
    # *** 修改：使用 /records/search 端点和 POST 方法 ***
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/search"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    # *** 修改：设置 page_size=1 并使用 POST ***
    payload = {"page_size": 1}

    print(f"      > 正在通过搜索获取表格 {table_id} 的记录总数 (page_size=1)...")
    try:
        # print(f"url: {url}")
        # print(f"headers: {headers}")
        # print(f"payload: {payload}")
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") == 0:
            # *** 修改：从 data.total 获取记录数 ***
            total_count = data.get("data", {}).get("total")
            if total_count is not None:
                try:
                    count = int(total_count)
                    print(f"      ✅ 获取到记录总数: {count}")
                    return count
                except (ValueError, TypeError):
                    print(
                        f"      ⚠️ 获取到 total 字段 ({total_count}) 但无法转换为整数。"
                    )
                    return None
            else:
                print(
                    f"      ⚠️ 未能在响应数据中找到 'total' 字段。响应: {data.get('data')}"
                )
                return None
        else:
            error_code = data.get("code")
            error_msg = data.get("msg", "未知错误")
            print(
                f"      ❌ 获取表格 {table_id} 记录数失败 (API Code: {error_code}): {error_msg}"
            )
            return None

    except requests.exceptions.Timeout:
        print(f"      ❌ 请求表格 {table_id} 记录数超时。")
        return None
    except requests.exceptions.HTTPError as http_err:
        # 捕获 HTTP 错误以打印更详细的信息
        print(f"      ❌ 请求表格 {table_id} 记录数时发生 HTTP 错误: {http_err}")
        if http_err.response is not None:
            print(f"         Response status: {http_err.response.status_code}")
            try:
                print(f"         Response body: {http_err.response.text}")
            except Exception:
                pass
        return None
    except requests.exceptions.RequestException as e:
        print(f"      ❌ 请求表格 {table_id} 记录数时发生网络错误: {e}")
        return None
    except Exception as e:
        print(f"      ❌ 获取表格 {table_id} 记录数时发生未知错误: {e}")
        return None


# === 新增：写入操作 ===


def batch_delete_records(
    app_token: str, table_id: str, record_ids: List[str], app_id: str, app_secret: str
) -> dict:
    """批量删除飞书多维表格中的记录。"""
    results = {"success_count": 0, "error_count": 0, "errors": []}
    if not record_ids:
        print("   [Feishu Delete] 无记录需要删除。")
        return results

    print(
        f"   [Feishu Delete] 准备删除表格 {table_id} 中的 {len(record_ids)} 条记录..."
    )
    try:
        access_token = get_access_token(app_id, app_secret)
        if not access_token:
            raise Exception("获取 Access Token 失败")

        BASE_URL = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_delete"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # 飞书批量删除限制未知，但通常批量接口有大小限制，先假设一个值，例如 500
        # TODO: 查阅飞书文档确认 batch_delete 的具体限制
        BATCH_DELETE_SIZE = 500

        for i in range(0, len(record_ids), BATCH_DELETE_SIZE):
            batch_ids = record_ids[i : i + BATCH_DELETE_SIZE]
            payload = {"records": batch_ids}
            print(
                f"      > 正在删除批次 {i // BATCH_DELETE_SIZE + 1} (共 {len(batch_ids)} 条)..."
            )

            try:
                response = requests.post(
                    BASE_URL, headers=headers, json=payload, timeout=60
                )
                response.raise_for_status()
                data = response.json()

                if data.get("code") == 0:
                    # 检查详细删除结果 (如果API提供)
                    deleted_count_in_batch = len(batch_ids)  # 假设成功则全部删除
                    # TODO: 检查飞书 batch_delete 的响应体，看是否包含更详细的成功/失败信息
                    # 示例：failed_records = data.get('data', {}).get('failures', [])
                    # deleted_count_in_batch = len(batch_ids) - len(failed_records)
                    # errors_in_batch = [f"ID {f.get('record_id')}: {f.get('error_message', '未知错误')}" for f in failed_records]

                    print(
                        f"         批次删除成功 (API返回码0)，处理 {deleted_count_in_batch} 条。"
                    )
                    results["success_count"] += deleted_count_in_batch
                    # results["errors"].extend(errors_in_batch)
                    # results["error_count"] += len(errors_in_batch)
                else:
                    error_code = data.get("code")
                    error_msg = data.get("msg", "未知错误")
                    print(f"      ❌ 批次删除失败: Code={error_code}, Msg={error_msg}")
                    results["error_count"] += len(batch_ids)  # 假设整个批次失败
                    results["errors"].append(
                        f"批次删除API错误 (Code: {error_code}): {error_msg} (影响 {len(batch_ids)} 条记录)"
                    )

            except requests.exceptions.Timeout:
                print(f"      ❌ 批次删除请求超时。")
                results["error_count"] += len(batch_ids)
                results["errors"].append(
                    f"批次删除请求超时 (影响 {len(batch_ids)} 条记录)"
                )
            except requests.exceptions.RequestException as req_err:
                print(f"      ❌ 批次删除请求网络错误: {req_err}")
                results["error_count"] += len(batch_ids)
                results["errors"].append(
                    f"批次删除网络错误: {req_err} (影响 {len(batch_ids)} 条记录)"
                )

            # 稍微延时避免触发流控
            if i + BATCH_DELETE_SIZE < len(record_ids):
                time.sleep(0.3)

    except Exception as e:
        print(f"   ❌ 批量删除过程中发生意外错误: {e}")
        results["error_count"] = len(record_ids)  # 标记所有为失败
        results["errors"].append(f"批量删除主流程错误: {e}")

    print(
        f"   [Feishu Delete] 删除操作完成。成功: {results['success_count']}, 失败: {results['error_count']}"
    )
    return results


def batch_update_records(
    app_token: str,
    table_id: str,
    records_to_update: List[Dict],
    app_id: str,
    app_secret: str,
) -> dict:
    """批量更新飞书多维表格中的记录。
    Args:
        records_to_update: 字典列表，每个字典包含 'record_id' 和 'fields' (要更新的字段键值对)。
                           例如: [{'record_id': 'recxxxx', 'fields': {'字段A': '新值A'}}]
    """
    results = {"success_count": 0, "error_count": 0, "errors": []}
    if not records_to_update:
        print("   [Feishu Update] 无记录需要更新。")
        return results

    print(
        f"   [Feishu Update] 准备更新表格 {table_id} 中的 {len(records_to_update)} 条记录..."
    )
    try:
        access_token = get_access_token(app_id, app_secret)
        if not access_token:
            raise Exception("获取 Access Token 失败")

        BASE_URL = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # 飞书批量更新限制通常也是 500
        # TODO: 查阅飞书文档确认 batch_update 的具体限制和请求/响应格式
        BATCH_UPDATE_SIZE = 500

        # 新增：定义API保留字段黑名单
        BLACKLIST_FIELDS = [
            "table_id",
            "Table_ID",
            "tableid",
            "TableID",
            "tableID",
            "tableId",
            "record_id",
            "Record_ID",
            "recordid",
            "RecordID",
            "recordID",
            "recordId",
            "app_token",
            "App_Token",
            "apptoken",
            "AppToken",
            "appToken",
            "app_id",
            "App_ID",
            "appid",
            "AppID",
            "appID",
            "appId",
            "app_secret",
            "App_Secret",
            "appsecret",
            "AppSecret",
            "appSecret",
        ]
        print(f"   [Feishu Update] 应用字段黑名单过滤: {BLACKLIST_FIELDS}")

        # 新增：应用黑名单过滤
        filtered_records = []
        field_removal_count = 0

        for record in records_to_update:
            filtered_record = record.copy()
            filtered_fields = {}

            # 复制原始字段，排除黑名单字段
            for field_name, field_value in record.get("fields", {}).items():
                if field_name not in BLACKLIST_FIELDS:
                    filtered_fields[field_name] = field_value
                else:
                    field_removal_count += 1
                    print(f"      [Safety] 记录中移除黑名单字段: '{field_name}'")

            # 更新过滤后的字段
            filtered_record["fields"] = filtered_fields
            filtered_records.append(filtered_record)

        # 使用过滤后的记录替代原始记录
        records_to_update = filtered_records

        if field_removal_count > 0:
            print(
                f"   [Feishu Update] 安全过滤: 共移除 {field_removal_count} 个黑名单字段实例"
            )

        # 新增：获取表格字段元数据
        field_type_map = get_table_fields_metadata(access_token, app_token, table_id)
        phone_field_type = field_type_map.get("电话")
        if phone_field_type == 2:  # 2=数字类型
            print(
                "   [Feishu Update] 检测到'电话'字段为数字类型，将自动转换为数字格式进行同步。"
            )
            for record in records_to_update:
                fields = record.get("fields", {})
                phone_val = fields.get("电话")
                if phone_val is not None:
                    try:
                        phone_num = int("".join(filter(str.isdigit, str(phone_val))))
                        fields["电话"] = phone_num
                    except Exception as e:
                        print(
                            f"      [Feishu Update] 电话字段转换失败: {phone_val} -> {e}"
                        )
                        fields["电话"] = None

        for i in range(0, len(records_to_update), BATCH_UPDATE_SIZE):
            batch_records = records_to_update[i : i + BATCH_UPDATE_SIZE]
            # 构造请求体，格式通常是 {"records": [...]}，其中每个元素包含 record_id 和 fields
            payload = {"records": batch_records}
            print(
                f"      > 正在更新批次 {i // BATCH_UPDATE_SIZE + 1} (共 {len(batch_records)} 条)..."
            )

            try:
                response = requests.post(
                    BASE_URL, headers=headers, json=payload, timeout=120
                )  # 更新操作可能耗时更长，增加超时
                response.raise_for_status()
                data = response.json()

                if data.get("code") == 0:
                    # 检查详细更新结果
                    # 飞书 batch_update 响应通常包含一个 records 列表，其中每个记录可能成功或失败
                    updated_records_info = data.get("data", {}).get("records", [])
                    success_in_batch = 0
                    errors_in_batch = []
                    # 假设成功响应的 updated_records_info 列表长度与请求批次相同
                    if len(updated_records_info) == len(batch_records):
                        success_in_batch = len(batch_records)  # 简单假设全部成功
                        # 更精确的方式是检查 updated_records_info 中每个元素的状态，如果API提供的话
                    else:
                        # 如果响应记录数不匹配，可能部分成功或全部失败？需要文档确认
                        # 暂时按成功数估算 (需要API文档确认响应格式)
                        success_in_batch = len(updated_records_info)
                        print(
                            f"       ⚠️ 更新响应记录数({len(updated_records_info)})与请求数({len(batch_records)})不符，成功计数可能不准。"
                        )

                    results["success_count"] += success_in_batch
                    # TODO: 解析可能的失败详情，填充 errors_in_batch
                    # results["error_count"] += len(batch_records) - success_in_batch
                    # results["errors"].extend(errors_in_batch)
                    print(
                        f"         批次更新完成 (API返回码0)，估算成功 {success_in_batch} 条。"
                    )
                else:
                    error_code = data.get("code")
                    error_msg = data.get("msg", "未知错误")
                    print(f"      ❌ 批次更新失败: Code={error_code}, Msg={error_msg}")
                    results["error_count"] += len(batch_records)
                    results["errors"].append(
                        f"批次更新API错误 (Code: {error_code}): {error_msg} (影响 {len(batch_records)} 条记录)"
                    )

            except requests.exceptions.Timeout:
                print(f"      ❌ 批次更新请求超时。")
                results["error_count"] += len(batch_records)
                results["errors"].append(
                    f"批次更新请求超时 (影响 {len(batch_records)} 条记录)"
                )
            except requests.exceptions.RequestException as req_err:
                print(f"      ❌ 批次更新请求网络错误: {req_err}")
                results["error_count"] += len(batch_records)
                results["errors"].append(
                    f"批次更新网络错误: {req_err} (影响 {len(batch_records)} 条记录)"
                )

            # 延时
            if i + BATCH_UPDATE_SIZE < len(records_to_update):
                time.sleep(0.3)

    except Exception as e:
        print(f"   ❌ 批量更新过程中发生意外错误: {e}")
        results["error_count"] = len(records_to_update)
        results["errors"].append(f"批量更新主流程错误: {e}")

    print(
        f"   [Feishu Update] 更新操作完成。成功: {results['success_count']}, 失败: {results['error_count']}"
    )
    return results


def batch_add_records(
    app_token: str,
    table_id: str,
    records_to_add: List[Dict],  # List of dicts, each containing {'fields': {...}}
    app_id: str,
    app_secret: str,
) -> dict:
    """批量添加新记录到飞书多维表格。"""
    results = {"success_count": 0, "error_count": 0, "errors": []}
    if not records_to_add:
        print("   [Feishu Add] 无记录需要新增。")
        return results

    print(f"   [Feishu Add] 准备新增 {len(records_to_add)} 条记录到表格 {table_id}...")
    try:
        access_token = get_access_token(app_id, app_secret)
        if not access_token:
            raise Exception("获取 Access Token 失败")

        BASE_URL = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_create"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        BATCH_CREATE_SIZE = 500

        # 首先输出所有记录中使用的字段名统计
        all_field_names = set()
        field_usage_count = {}
        for record in records_to_add:
            record_fields = record.get("fields", {})
            for field_name in record_fields:
                all_field_names.add(field_name)
                field_usage_count[field_name] = field_usage_count.get(field_name, 0) + 1

        print(
            f"   [DEBUG] 找到 {len(all_field_names)} 个不同的字段名在 {len(records_to_add)} 条记录中"
        )
        print(f"   [DEBUG] 字段使用频率统计 (前10个):")
        sorted_fields = sorted(
            field_usage_count.items(), key=lambda x: x[1], reverse=True
        )
        for field, count in sorted_fields[:10]:
            print(
                f"      - '{field}': 出现在 {count} 条记录中 ({count/len(records_to_add)*100:.1f}%)"
            )

        # 检查是否存在可能导致问题的字段
        # 新增：扩展黑名单字段
        BLACKLIST_FIELDS = [
            "table_id",
            "Table_ID",
            "tableid",
            "TableID",
            "tableID",
            "tableId",
            "record_id",
            "Record_ID",
            "recordid",
            "RecordID",
            "recordID",
            "recordId",
            "app_token",
            "App_Token",
            "apptoken",
            "AppToken",
            "appToken",
            "app_id",
            "App_ID",
            "appid",
            "AppID",
            "appID",
            "appId",
            "app_secret",
            "App_Secret",
            "appsecret",
            "AppSecret",
            "appSecret",
        ]
        print(f"   [Feishu Add] 应用字段黑名单过滤: {BLACKLIST_FIELDS}")

        # 新增：应用黑名单过滤
        filtered_records = []
        field_removal_count = 0

        for record in records_to_add:
            filtered_record = record.copy()
            filtered_fields = {}

            # 复制原始字段，排除黑名单字段
            for field_name, field_value in record.get("fields", {}).items():
                if field_name not in BLACKLIST_FIELDS:
                    filtered_fields[field_name] = field_value
                else:
                    field_removal_count += 1
                    print(f"      [Safety] 记录中移除黑名单字段: '{field_name}'")

            # 更新过滤后的字段
            filtered_record["fields"] = filtered_fields
            filtered_records.append(filtered_record)

        # 使用过滤后的记录替代原始记录
        records_to_add = filtered_records

        if field_removal_count > 0:
            print(
                f"   [Feishu Add] 安全过滤: 共移除 {field_removal_count} 个黑名单字段实例"
            )

        # 新增：获取表格字段元数据
        field_type_map = get_table_fields_metadata(access_token, app_token, table_id)
        phone_field_type = field_type_map.get("电话")
        if phone_field_type == 2:  # 2=数字类型
            print(
                "   [Feishu Add] 检测到'电话'字段为数字类型，将自动转换为数字格式进行同步。"
            )
            for record in records_to_add:
                fields = record.get("fields", {})
                phone_val = fields.get("电话")
                if phone_val is not None:
                    # 尝试只保留数字字符并转为int
                    try:
                        phone_num = int("".join(filter(str.isdigit, str(phone_val))))
                        fields["电话"] = phone_num
                    except Exception as e:
                        print(
                            f"      [Feishu Add] 电话字段转换失败: {phone_val} -> {e}"
                        )
                        fields["电话"] = None

        for i in range(0, len(records_to_add), BATCH_CREATE_SIZE):
            batch_records_payload = records_to_add[i : i + BATCH_CREATE_SIZE]
            batch_number = i // BATCH_CREATE_SIZE + 1
            payload = {"records": batch_records_payload}

            # 详细记录每个批次的信息
            print(
                f"      > 正在新增批次 {batch_number} (共 {len(batch_records_payload)} 条)..."
            )

            # 如果是第一批或最后一批，详细记录更多信息
            if batch_number == 1 or batch_number * BATCH_CREATE_SIZE >= len(
                records_to_add
            ):
                print(f"      [DEBUG] 批次 {batch_number} 详细信息:")
                # 获取第一条和最后一条记录的字段列表
                first_record = (
                    batch_records_payload[0] if batch_records_payload else None
                )
                last_record = (
                    batch_records_payload[-1] if batch_records_payload else None
                )

                if first_record:
                    first_fields = first_record.get("fields", {})
                    print(
                        f"      [DEBUG] 第一条记录包含 {len(first_fields)} 个字段: {list(first_fields.keys())}"
                    )

                if last_record and last_record != first_record:
                    last_fields = last_record.get("fields", {})
                    print(
                        f"      [DEBUG] 最后一条记录包含 {len(last_fields)} 个字段: {list(last_fields.keys())}"
                    )

                # 再次检查是否存在异常字段（过滤后应该没有了）
                problem_records = []
                for idx, record in enumerate(batch_records_payload):
                    record_fields = record.get("fields", {})
                    for prob_field in BLACKLIST_FIELDS:
                        if prob_field in record_fields:
                            problem_records.append((idx, prob_field))

                if problem_records:
                    print(
                        f"      [WARNING] 在批次 {batch_number} 中仍然发现 {len(problem_records)} 条记录包含可能导致问题的字段 (过滤失败?):"
                    )
                    for idx, field in problem_records[:5]:  # 只显示前5个
                        record = batch_records_payload[idx]
                        print(
                            f"         - 记录 #{idx}: 包含字段 '{field}', 值: {record['fields'].get(field)}"
                        )
                    if len(problem_records) > 5:
                        print(
                            f"         - ... 以及其他 {len(problem_records)-5} 条记录"
                        )

            try:
                # 添加请求详情日志
                print(f"      [DEBUG] 发送请求到 {BASE_URL}")
                print(f"      [DEBUG] 请求头: {headers}")
                if len(batch_records_payload) > 0:
                    sample_record = batch_records_payload[0]
                    print(
                        f"      [DEBUG] 样本记录字段: {list(sample_record.get('fields', {}).keys())}"
                    )

                response = requests.post(
                    BASE_URL, headers=headers, json=payload, timeout=120
                )

                # 记录响应状态和响应头
                print(f"      [DEBUG] 响应状态码: {response.status_code}")
                print(f"      [DEBUG] 响应头: {dict(response.headers)}")

                # 记录完整响应内容
                try:
                    data = response.json()
                    # print(f"      [DEBUG] 响应内容: {data}")

                    if data.get("code") == 0:
                        added_records_info = data.get("data", {}).get("records", [])
                        success_in_batch = len(added_records_info)

                        # 理论上 batch_create 在 code=0 时，响应的 records 列表长度应与请求批次一致
                        # 但为保险起见，仍以响应中的记录数为准。
                        if success_in_batch != len(batch_records_payload):
                            print(
                                f"       ⚠️ 新增响应记录数({success_in_batch})与请求数({len(batch_records_payload)})不符，可能部分失败，请检查飞书后台。"
                            )
                            # 记录一个通用错误，因为无法确定哪些失败了
                            results["errors"].append(
                                f"批次 {batch_number}: 新增响应记录数与请求数不符"
                            )
                            results["error_count"] += (
                                len(batch_records_payload) - success_in_batch
                            )

                        results["success_count"] += success_in_batch
                        print(
                            f"         批次新增完成 (API Code 0)，成功 {success_in_batch} 条。"
                        )
                    else:
                        # *** 修改：提取更详细的错误信息 ***
                        error_code = data.get("code")
                        error_msg = data.get("msg", "未知错误")
                        detailed_error = data.get("error", {}).get(
                            "message", ""
                        )  # 尝试获取详细错误
                        log_id = data.get("error", {}).get("log_id", "N/A")
                        print(
                            f"      ❌ 批次新增失败: Code={error_code}, Msg={error_msg}"
                        )
                        if detailed_error:
                            print(f"         详细错误: {detailed_error}")
                        print(f"         Log ID: {log_id}")

                        # 分析错误信息中是否包含字段名相关的错误，如果有，查看是哪个字段导致的问题
                        if (
                            "field_name not found" in detailed_error.lower()
                            or "fields." in detailed_error
                        ):
                            print(f"      [ERROR] 检测到字段名相关错误!")
                            if "fields." in detailed_error:
                                # 尝试从错误消息中提取出有问题的字段名
                                import re

                                field_matches = re.findall(
                                    r"fields\.([^\s'\"\.]+)", detailed_error
                                )
                                if field_matches:
                                    problem_field = field_matches[0]
                                    print(
                                        f"      [ERROR] 可能的问题字段: '{problem_field}'"
                                    )

                                    # 新增：将问题字段添加到黑名单
                                    if problem_field not in BLACKLIST_FIELDS:
                                        BLACKLIST_FIELDS.append(problem_field)
                                        print(
                                            f"      [SAFETY] 已将字段 '{problem_field}' 添加到黑名单"
                                        )

                                    # 检查这个字段在记录中的分布情况
                                    records_with_field = [
                                        idx
                                        for idx, rec in enumerate(batch_records_payload)
                                        if problem_field in rec.get("fields", {})
                                    ]
                                    if records_with_field:
                                        print(
                                            f"      [ERROR] 该字段出现在批次的 {len(records_with_field)} 条记录中，索引: {records_with_field[:5]}..."
                                        )

                                        # 显示第一条包含问题字段的记录的完整内容
                                        if records_with_field:
                                            problem_record = batch_records_payload[
                                                records_with_field[0]
                                            ]
                                            print(
                                                f"      [ERROR] 问题记录示例: {problem_record}"
                                            )

                        results["error_count"] += len(
                            batch_records_payload
                        )  # 假设整个批次失败
                        error_log_entry = (
                            f"批次新增API错误 (Code: {error_code}, Msg: {error_msg}"
                            + (f", Detail: {detailed_error}" if detailed_error else "")
                            + f", LogID: {log_id}) (影响 {len(batch_records_payload)} 条记录)"
                        )
                        results["errors"].append(error_log_entry)
                except Exception as json_err:
                    print(f"      [ERROR] 解析响应JSON时出错: {json_err}")
                    print(
                        f"      [ERROR] 原始响应内容: {response.text[:500]}..."
                    )  # 只显示前500个字符

            except requests.exceptions.Timeout:
                print(f"      ❌ 批次新增请求超时。")
                results["error_count"] += len(batch_records_payload)
                results["errors"].append(
                    f"批次新增请求超时 (影响 {len(batch_records_payload)} 条记录)"
                )
            except requests.exceptions.HTTPError as http_err:
                # 处理 HTTP 层面的错误 (非业务错误 code)
                print(f"      ❌ 批次新增请求发生 HTTP 错误: {http_err}")
                results["error_count"] += len(batch_records_payload)
                error_detail = f"HTTP {http_err.response.status_code}"
                try:
                    error_detail += (
                        f": {http_err.response.text[:200]}..."  # 显示部分响应体
                    )
                except Exception:
                    pass
                results["errors"].append(
                    f"批次新增HTTP错误 {error_detail} (影响 {len(batch_records_payload)} 条记录)"
                )
            except requests.exceptions.RequestException as req_err:
                print(f"      ❌ 批次新增请求网络错误: {req_err}")
                results["error_count"] += len(batch_records_payload)
                results["errors"].append(
                    f"批次新增网络错误: {req_err} (影响 {len(batch_records_payload)} 条记录)"
                )
            except Exception as generic_err:
                # 捕获其他可能的错误 (如 JSON 解析失败等)
                print(f"      ❌ 处理批次新增响应时发生错误: {generic_err}")
                results["error_count"] += len(batch_records_payload)
                results["errors"].append(
                    f"处理批次新增响应错误: {generic_err} (影响 {len(batch_records_payload)} 条记录)"
                )

            # 延时
            if i + BATCH_CREATE_SIZE < len(records_to_add):
                time.sleep(0.3)

    except Exception as e:
        print(f"   ❌ 批量新增过程中发生意外错误: {e}")
        results["error_count"] = len(records_to_add)  # 标记所有为失败
        results["errors"].append(f"批量新增主流程错误: {e}")

    print(
        f"   [Feishu Add] 新增操作完成。成功: {results['success_count']}, 失败: {results['error_count']}"
    )
    # 打印详细错误信息
    if results["errors"]:
        print("      详细错误列表:")
        for err_item in results["errors"]:
            print(f"         - {err_item}")

    return results


# 新增：获取表格字段元数据


def get_table_fields_metadata(access_token, app_token, table_id):
    """获取飞书表格字段元数据，返回字段名到类型的映射。"""
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0:
            fields = data.get("data", {}).get("items", [])
            # 返回字段名到类型的映射（type: 1=文本, 2=数字, 3=单选, ...）
            return {
                f["field_name"]: f["type"]
                for f in fields
                if "field_name" in f and "type" in f
            }
        else:
            print(f"   [Feishu Meta] 获取字段元数据失败: {data.get('msg')}")
            return {}
    except Exception as e:
        print(f"   [Feishu Meta] 获取字段元数据异常: {e}")
        return {}
