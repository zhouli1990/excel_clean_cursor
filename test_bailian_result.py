#!/usr/bin/env python3

import os
import json
import traceback
import pandas as pd
import processor
import postprocessor


def test_bailian_result_parsing():
    """测试百炼API结果解析功能"""
    print("\n=== 测试百炼API结果解析 ===")

    # 使用现有的结果文件
    result_file = "uploads/67c96a39-a281-4708-9049-4ec862d141d6/batch_result.jsonl"

    if not os.path.exists(result_file):
        print(f"❌ 测试文件不存在: {result_file}")
        return

    print(f"使用测试文件: {result_file}")

    # 读取结果文件的第一行
    with open(result_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    try:
        # 解析JSON行
        record = json.loads(first_line)
        custom_id = record.get("custom_id", "未知ID")
        print(f"批次ID: {custom_id}")

        # 提取LLM返回的内容
        content = None
        if "response" in record and "body" in record["response"]:
            content = (
                record["response"]["body"]["choices"][0]
                .get("message", {})
                .get("content", "")
            )
        elif "body" in record and "choices" in record["body"]:
            content = record["body"]["choices"][0].get("message", {}).get("content", "")

        if not content:
            print("❌ 结果中没有找到content字段")
            return

        # 打印原始content的前100个字符
        print(f"原始content (前100字符): {content[:100]}...")

        # 使用修复后的逻辑来解析content
        content = content.strip()

        # 测试方式1: 直接解析
        if content.startswith("[") and content.endswith("]"):
            try:
                print("\n方式1: 尝试直接解析JSON数组...")
                llm_results = json.loads(content)
                if isinstance(llm_results, list):
                    print(f"✅ 直接解析成功，得到 {len(llm_results)} 个结果项")
                    if len(llm_results) > 0:
                        print(
                            f"  第一项: {json.dumps(llm_results[0], ensure_ascii=False)[:100]}..."
                        )
                else:
                    print(f"❌ 解析结果不是数组: {type(llm_results)}")
            except Exception as e:
                print(f"❌ 直接解析失败: {e}")

                # 测试方式2: 解码Unicode后解析
                try:
                    print("\n方式2: 尝试解码Unicode转义字符后解析...")
                    cleaned_content = content.encode("utf-8").decode("unicode_escape")
                    llm_results = json.loads(cleaned_content)
                    if isinstance(llm_results, list):
                        print(
                            f"✅ Unicode解码后解析成功，得到 {len(llm_results)} 个结果项"
                        )
                        if len(llm_results) > 0:
                            print(
                                f"  第一项: {json.dumps(llm_results[0], ensure_ascii=False)[:100]}..."
                            )
                    else:
                        print(f"❌ 解析结果不是数组: {type(llm_results)}")
                except Exception as e2:
                    print(f"❌ Unicode解码后解析仍失败: {e2}")
        else:
            # 测试方式3: 提取JSON数组部分
            try:
                print("\n方式3: 尝试提取并解析JSON数组部分...")
                start_idx = content.find("[")
                end_idx = content.rfind("]")
                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    print("❌ 无法找到有效的JSON数组范围")
                else:
                    json_part = content[start_idx : end_idx + 1]
                    llm_results = json.loads(json_part)
                    if isinstance(llm_results, list):
                        print(f"✅ 提取并解析成功，得到 {len(llm_results)} 个结果项")
                        if len(llm_results) > 0:
                            print(
                                f"  第一项: {json.dumps(llm_results[0], ensure_ascii=False)[:100]}..."
                            )
                    else:
                        print(f"❌ 提取的内容不是有效的JSON数组: {type(llm_results)}")
            except Exception as e:
                print(f"❌ 尝试提取JSON数组部分失败: {e}")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        print(traceback.format_exc())

    print("\n测试完成")


def test_bailian_process():
    print("开始测试百炼批处理结果处理...")

    # 批次ID，从前面的测试中获取
    batch_id = "batch_96e97090-375d-45aa-b5c2-35b73ac4aaba"

    # 任务ID
    task_id = "test_task_20250509"

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

    # 创建输出目录
    output_dir = f"outputs/{task_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 步骤1: 检查百炼批处理任务状态
    print("\n步骤1: 检查百炼批处理任务状态")
    status_info = processor.check_bailian_job_status(batch_id, config)
    print(f"批处理状态: {status_info}")

    # 步骤2: 下载和处理百炼结果
    print("\n步骤2: 下载和处理百炼结果")
    try:
        processed_df = processor.download_and_process_bailian_results(
            batch_id, original_df, task_id, config  # 空的原始DataFrame
        )
        if processed_df is not None:
            print(f"百炼结果处理完成，获得 DataFrame，形状: {processed_df.shape}")
            print(f"列: {processed_df.columns.tolist()}")
        else:
            print("警告: 百炼结果处理后返回了空的 DataFrame")
    except Exception as e:
        print(f"错误: 下载和处理百炼结果时出错: {e}")
        import traceback

        traceback.print_exc()
        return

    # 步骤3: 应用后处理逻辑
    print("\n步骤3: 应用后处理逻辑")
    if processed_df is not None and not processed_df.empty:
        try:
            # 应用后处理步骤
            print(f"应用后处理步骤，DataFrame大小: {len(processed_df)} 行...")
            processed_df = postprocessor.apply_post_processing(processed_df, config)

            # 步骤4: 创建多Sheet页Excel文件
            print("\n步骤4: 创建多Sheet页Excel文件")
            final_output_file = os.path.join(output_dir, f"final_test.xlsx")
            print(f"创建多Sheet页Excel文件: {final_output_file}")
            postprocessor.create_multi_sheet_excel(
                processed_df, final_output_file, config
            )

            print(f"结果已保存到 {final_output_file}")
        except Exception as e:
            print(f"错误: 后处理或创建Excel文件时出错: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("错误: 无法应用后处理，DataFrame为空")


if __name__ == "__main__":
    test_bailian_result_parsing()
    test_bailian_process()
