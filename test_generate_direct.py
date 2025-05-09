import os
import pandas as pd
import uuid
import postprocessor
from datetime import datetime


def generate_direct_result():
    """
    直接加载Excel文件并生成最终结果文件，绕过百炼处理过程
    """
    print("开始直接生成结果文件...")

    # 输入文件路径
    input_file = "/Users/zhouli/Downloads/250428-活动-2025年4月CDIE名单2.xlsx"

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        return

    # 创建任务ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 创建输出目录
    output_dir = f"outputs/{task_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 配置
    config = {
        "feishu_config": {
            "COMPANY_NAME_COLUMN": "企业名称",
            "PHONE_NUMBER_COLUMN": "电话",
            "RELATED_COMPANY_COLUMN_NAME": "关联公司名称(LLM)",
            "REMARK_COLUMN_NAME": "备注",
        }
    }

    # 步骤1: 读取Excel文件
    print(f"步骤1: 读取Excel文件: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"成功读取Excel文件，获得 {len(df)} 行数据")
        print(f"列: {df.columns.tolist()}")
    except Exception as e:
        print(f"错误: 读取Excel文件时出错: {e}")
        return

    # 步骤2: 添加必要的列和映射列名
    print("步骤2: 添加必要的列和映射列名")
    # 添加 local_row_id 列
    df["local_row_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # 添加 来源 列
    source_file_name = os.path.basename(input_file)
    df["来源"] = source_file_name

    # 映射列名
    column_mapping = {"公司名称": "企业名称", "联系电话": "电话"}

    # 复制列
    for src_col, dst_col in column_mapping.items():
        if src_col in df.columns:
            df[dst_col] = df[src_col]
            print(f"复制列: {src_col} -> {dst_col}")

    # 确保有备注列和关联公司列
    if "备注" not in df.columns:
        df["备注"] = ""
    if "关联公司名称(LLM)" not in df.columns:
        df["关联公司名称(LLM)"] = ""

    # 添加空的record_id列，用于后处理
    df["record_id"] = ""

    # 步骤3: 应用后处理逻辑
    print("步骤3: 应用后处理逻辑")
    try:
        processed_df = postprocessor.apply_post_processing(df, config)
        print(f"后处理完成，DataFrame大小: {len(processed_df)} 行")
    except Exception as e:
        print(f"错误: 后处理时出错: {e}")
        import traceback

        traceback.print_exc()
        return

    # 步骤4: 创建多Sheet页Excel文件
    print("步骤4: 创建多Sheet页Excel文件")
    try:
        final_output_file = os.path.join(output_dir, f"final_{task_id}.xlsx")
        print(f"创建多Sheet页Excel文件: {final_output_file}")
        postprocessor.create_multi_sheet_excel(processed_df, final_output_file, config)
        print(f"结果已保存到: {final_output_file}")
    except Exception as e:
        print(f"错误: 创建Excel文件时出错: {e}")
        import traceback

        traceback.print_exc()
        return

    # 创建一个简单的consolidated文件，作为备份
    consolidated_file = os.path.join(output_dir, f"consolidated_{task_id}.xlsx")
    df.to_excel(consolidated_file, index=False)
    print(f"原始数据已保存到: {consolidated_file}")

    print(f"直接生成结果文件完成，文件保存在: {output_dir}")

    # 返回任务ID和结果文件路径
    return {
        "task_id": task_id,
        "final_file": final_output_file,
        "consolidated_file": consolidated_file,
    }


if __name__ == "__main__":
    generate_direct_result()
