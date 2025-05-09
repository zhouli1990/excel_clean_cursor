import os
import sys
import pandas as pd
import processor


def download_batch_result():
    # 批次ID
    batch_id = "batch_6d28cb3f-dca2-40c0-a521-cd50b35fdf2a"

    # 任务ID
    task_id = "task_20250509_143852_41bfc3e0"

    # 创建一个空的原始DataFrame
    original_df = pd.DataFrame()

    # 基本配置
    config = {
        "llm_config": {
            "API_TIMEOUT": 180,
            "BAILIAN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "BAILIAN_BATCH_ENDPOINT": "/v1/chat/completions",
            "BAILIAN_COMPLETION_WINDOW": "24h",
            "BAILIAN_MODEL_NAME": "qwen-turbo-latest",
            "BATCH_SIZE": 50,
            "DASHSCOPE_API_KEY": "sk-ef56a7163b2b4eafb87eb1bdd579bb6f",
        },
        "feishu_config": {
            "COMPANY_NAME_COLUMN": "企业名称",
            "PHONE_NUMBER_COLUMN": "电话",
            "RELATED_COMPANY_COLUMN_NAME": "关联公司名称(LLM)",
            "REMARK_COLUMN_NAME": "备注",
        },
    }

    # 步骤1: 检查百炼批处理任务状态
    print("步骤1: 检查百炼批处理任务状态")
    status_info = processor.check_bailian_job_status(batch_id, config)
    print(f"批处理状态: {status_info}")

    # 步骤2: 下载和处理百炼结果
    print("步骤2: 下载和处理百炼结果")
    try:
        processed_df = processor.download_and_process_bailian_results(
            batch_id, original_df, task_id, config  # 空的原始DataFrame
        )
        if processed_df is not None:
            print(f"百炼结果处理完成，获得 DataFrame，形状: {processed_df.shape}")
            print(f"列: {processed_df.columns.tolist()}")

            # 保存处理结果
            output_dir = f"outputs/download_test"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "downloaded_result.xlsx")
            processed_df.to_excel(output_file, index=False)
            print(f"结果已保存到: {output_file}")
        else:
            print("警告: 百炼结果处理后返回了空的 DataFrame")
    except Exception as e:
        print(f"错误: 下载和处理百炼结果时出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    download_batch_result()
