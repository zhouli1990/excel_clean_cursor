# -*- coding: utf-8 -*-
import pandas as pd
import traceback
from typing import List, Dict, Any, Union


def compare_dataframes_hybrid(
    df_final: pd.DataFrame,
    df_edited: pd.DataFrame,
    id_column: str = "record_id",
    local_id_column: str = "local_row_id",
    columns_to_compare: list[str] | None = None,
) -> dict:
    """
    Compares two DataFrames using a dedicated local ID column as the primary key.

    Args:
        df_final: The original DataFrame (e.g., from the final processing step).
        df_edited: The user-edited DataFrame.
        id_column: The column containing an external identifier (e.g., Feishu record_id). Used for context.
        local_id_column: The column containing the unique local identifier for *all* rows. This is the primary key for comparison.
        columns_to_compare: List of columns to check for modifications. If None, compares all shared columns except id_column and local_id_column.

    Returns:
        A dictionary containing differences:
        {
            'summary': {'added': ..., 'deleted': ..., 'modified': ...},
            'diffs': [
                {
                    'local_row_id': 'local_uuid_abc',
                    'record_id': 'rec123' or None,
                    'status': 'Modified', 'data': {...}, 'changes': {...}
                },
                # ...
            ],
            'columns': list_of_all_columns_in_diff_data
        }
    """
    all_diffs: List[Dict[str, Any]] = []
    print(f"开始比较 (v2)，主键: '{local_id_column}', 外部ID列: '{id_column}'")

    # --- 1. 预处理和验证 ---
    if (
        local_id_column not in df_final.columns
        or local_id_column not in df_edited.columns
    ):
        raise ValueError(
            f"关键列 '{local_id_column}' 必须存在于两个 DataFrame 中才能比较。"
        )

    # Ensure the external id_column exists, fill with None if missing
    if id_column not in df_final.columns:
        print(f"   警告: 外部 ID 列 '{id_column}' 不在 df_final 中，将添加空列。")
        df_final[id_column] = None
    if id_column not in df_edited.columns:
        print(f"   警告: 外部 ID 列 '{id_column}' 不在 df_edited 中，将添加空列。")
        df_edited[id_column] = None

    # Fill NaN/NaT in id_column with None for consistency
    df_final[id_column] = (
        df_final[id_column].astype(object).where(pd.notnull(df_final[id_column]), None)
    )
    df_edited[id_column] = (
        df_edited[id_column]
        .astype(object)
        .where(pd.notnull(df_edited[id_column]), None)
    )

    # Fill NaN/NaT in local_id_column as well and convert to string to be safe
    # It *shouldn't* have NaNs, but better safe than sorry.
    df_final[local_id_column] = (
        df_final[local_id_column].fillna("missing_local_id_").astype(str)
    )
    df_edited[local_id_column] = (
        df_edited[local_id_column].fillna("missing_local_id_").astype(str)
    )

    # Check for duplicate local_ids which would break the comparison
    if df_final[local_id_column].duplicated().any():
        duplicates = df_final[df_final[local_id_column].duplicated()][
            local_id_column
        ].unique()
        print(
            f"❌ 错误: df_final 中的 '{local_id_column}' 存在重复值: {duplicates}。无法进行比较。"
        )
        raise ValueError(f"df_final 中的 '{local_id_column}' 存在重复值。")
    if df_edited[local_id_column].duplicated().any():
        duplicates = df_edited[df_edited[local_id_column].duplicated()][
            local_id_column
        ].unique()
        print(
            f"❌ 错误: df_edited 中的 '{local_id_column}' 存在重复值: {duplicates}。无法进行比较。"
        )
        raise ValueError(f"df_edited 中的 '{local_id_column}' 存在重复值。")

    # Get the final list of columns present in df_final (our baseline)
    final_columns = list(df_final.columns)

    # Determine which columns to actually compare for modifications
    if columns_to_compare is None:
        # Default: compare all columns present in *both* dfs, excluding the key columns
        common_columns = list(df_final.columns.intersection(df_edited.columns))
        columns_to_compare_effective = [
            col for col in common_columns if col not in [local_id_column, id_column]
        ]
    else:
        # Use provided list, but filter out key columns
        columns_to_compare_effective = [
            col for col in columns_to_compare if col not in [local_id_column, id_column]
        ]
        # Also ensure these columns actually exist in both dataframes
        columns_to_compare_effective = [
            col
            for col in columns_to_compare_effective
            if col in df_final.columns and col in df_edited.columns
        ]

    print(f"   将比较以下列的内容变化: {columns_to_compare_effective}")

    # --- 2. 比较 (使用 compare_subset) ---
    try:
        all_diffs = compare_subset(
            df_old=df_final,
            df_new=df_edited,
            key_column_or_list=local_id_column,  # Use the local ID as the key
            columns_to_compare=columns_to_compare_effective,  # Pass the calculated list
            is_composite_key=False,  # It's a single column key
            id_column_for_context=id_column,  # Pass the Feishu ID column name for context
        )
        print(f"  比较完成，发现 {len(all_diffs)} 项差异。")
    except Exception as e:
        print(f"❌ 比较数据时出错: {e}")
        traceback.print_exc()
        # Return empty diffs or re-raise depending on desired behavior
        return {
            "summary": {"added": 0, "deleted": 0, "modified": 0},
            "diffs": [],
            "columns": final_columns,
        }

    # --- 3. 汇总结果 ---
    summary = {"added": 0, "deleted": 0, "modified": 0}
    for diff in all_diffs:
        # The status is now directly set by compare_subset
        status = diff.get("status")
        if status == "Added":
            summary["added"] += 1
        elif status == "Deleted":
            summary["deleted"] += 1
        elif status == "Modified":
            summary["modified"] += 1

    print(
        f"比较完成。总差异: Added={summary['added']}, Deleted={summary['deleted']}, Modified={summary['modified']}"
    )

    return {
        "summary": summary,
        "diffs": all_diffs,  # Diffs now contain local_row_id and record_id
        "columns": final_columns,  # Provide column info for rendering
    }


def compare_subset(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    key_column_or_list: Union[str, List[str]],
    columns_to_compare: List[str],  # Now required and pre-filtered
    is_composite_key: bool = False,  # Kept for signature consistency, but will be False from new hybrid func
    id_column_for_context: str | None = None,  # NEW: Pass Feishu ID col name
) -> List[Dict[str, Any]]:
    """Compares two DataFrames based on a key, identifying added, deleted, and modified rows."""
    diffs: List[Dict[str, Any]] = []
    # Simplified log message
    key_desc = (
        key_column_or_list
        if isinstance(key_column_or_list, str)
        else ", ".join(key_column_or_list)
    )
    print(f"    子集比较 (主键: '{key_desc}'): 旧({len(df_old)}) vs 新({len(df_new)})")

    if df_old.empty and df_new.empty:
        return diffs

    primary_key_cols = (
        [key_column_or_list]
        if isinstance(key_column_or_list, str)
        else key_column_or_list
    )
    primary_key_col = primary_key_cols[0]  # Since we expect single key now

    # --- 预处理键列 (Simplified: basic check done in caller) ---
    # Caller (compare_dataframes_hybrid) now ensures key column exists and is filled.
    # We still need to ensure it's string type for reliable indexing/comparison.
    df_old_indexed = df_old.copy()
    df_new_indexed = df_new.copy()
    df_old_indexed[primary_key_col] = df_old_indexed[primary_key_col].astype(str)
    df_new_indexed[primary_key_col] = df_new_indexed[primary_key_col].astype(str)

    # --- 设置索引 --- #
    # Caller ensures no duplicates in the primary key column.
    try:
        df_old_indexed = df_old_indexed.set_index(primary_key_col, drop=False)
        df_new_indexed = df_new_indexed.set_index(primary_key_col, drop=False)
    except KeyError as e:
        print(f"    ❌ 错误: 设置索引时主键列 '{primary_key_col}' 不存在: {e}")
        return diffs
    except Exception as e:
        print(f"    ❌ 错误: 设置索引时发生未知错误: {e}")
        traceback.print_exc()
        return diffs

    # --- 识别状态 ---
    old_keys = set(df_old_indexed.index)
    new_keys = set(df_new_indexed.index)

    added_keys = new_keys - old_keys
    deleted_keys = old_keys - new_keys
    common_keys = old_keys.intersection(new_keys)

    # --- 处理差异 ---

    # Added rows
    for key in added_keys:
        new_row_series = df_new_indexed.loc[key]
        row_data = new_row_series.fillna("").to_dict()
        diff_item = {
            primary_key_col: key,  # Use actual primary key col name
            "status": "Added",
            "data": row_data,
            "changes": {},
            # Include context ID if available
            id_column_for_context: (
                row_data.get(id_column_for_context) if id_column_for_context else None
            ),
            # Add original index from the NEW dataframe
            "original_index": new_row_series.name,  # .name holds the original index before set_index
        }
        # Remove None context ID if it was added
        if id_column_for_context and diff_item[id_column_for_context] is None:
            del diff_item[id_column_for_context]

        diffs.append(diff_item)

    # Deleted rows
    for key in deleted_keys:
        old_row_series = df_old_indexed.loc[key]
        row_data = old_row_series.fillna("").to_dict()
        diff_item = {
            primary_key_col: key,
            "status": "Deleted",
            "data": row_data,
            "changes": {},
            id_column_for_context: (
                row_data.get(id_column_for_context) if id_column_for_context else None
            ),
            # Add original index from the OLD dataframe
            "original_index": old_row_series.name,
        }
        if id_column_for_context and diff_item[id_column_for_context] is None:
            del diff_item[id_column_for_context]
        diffs.append(diff_item)

    # Modified rows
    if columns_to_compare:  # Only check modifications if there are columns to compare
        for key in common_keys:
            changes = {}
            old_row = df_old_indexed.loc[key]
            new_row = df_new_indexed.loc[key]

            for col in columns_to_compare:
                old_val = old_row.get(col)
                new_val = new_row.get(col)

                # Normalize NaN/None/empty strings for comparison
                old_val_norm = "" if pd.isna(old_val) or old_val == "" else str(old_val)
                new_val_norm = "" if pd.isna(new_val) or new_val == "" else str(new_val)

                if old_val_norm != new_val_norm:
                    changes[col] = {"old": old_val_norm, "new": new_val_norm}

            if changes:
                row_data = new_row.fillna(
                    ""
                ).to_dict()  # Show current data in 'data' field
                diff_item = {
                    primary_key_col: key,
                    "status": "Modified",
                    "data": row_data,
                    "changes": changes,
                    id_column_for_context: (
                        row_data.get(id_column_for_context)
                        if id_column_for_context
                        else None
                    ),
                    # Add original index from the NEW dataframe (as it exists in both)
                    "original_index": new_row.name,
                }
                if id_column_for_context and diff_item[id_column_for_context] is None:
                    del diff_item[id_column_for_context]
                diffs.append(diff_item)
    else:
        print("    跳过修改检测，因为没有可比较的列。")

    print(f"    子集比较完成: {len(diffs)} 项差异被记录。")
    return diffs
