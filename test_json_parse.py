#!/usr/bin/env python3

import json
import traceback


def test_json_parsing():
    """测试改进后的JSON解析功能"""
    print("\n=== 测试JSON解析功能 ===")

    # 测试案例1: 标准JSON数组
    test_case1 = """[{"公司名称":"测试公司1","联系人":"张三","职位":"经理","电话":"13800138000","行ID":"123"},
                     {"公司名称":"测试公司2","联系人":"李四","职位":"总监","电话":"13900139000","行ID":"456"}]"""

    # 测试案例2: 被Unicode转义的中文JSON数组
    test_case2 = """[{\"\\u4f01\\u4e1a\\u540d\\u79f0\":\"\\u6d4b\\u8bd5\\u516c\\u53f81\",\"\\u8054\\u7cfb\\u4eba\":\"\\u5f20\\u4e09\",\"\\u804c\\u4f4d\":\"\\u7ecf\\u7406\",\"\\u7535\\u8bdd\":\"13800138000\",\"\\u884cID\":\"123\"},
                     {\"\\u4f01\\u4e1a\\u540d\\u79f0\":\"\\u6d4b\\u8bd5\\u516c\\u53f82\",\"\\u8054\\u7cfb\\u4eba\":\"\\u674e\\u56db\",\"\\u804c\\u4f4d\":\"\\u603b\\u76d1\",\"\\u7535\\u8bdd\":\"13900139000\",\"\\u884cID\":\"456\"}]"""

    # 测试案例3: JSON数组嵌入在文本中
    test_case3 = """我返回的数据如下：
    [{"公司名称":"测试公司1","联系人":"张三","职位":"经理","电话":"13800138000","行ID":"123"},
     {"公司名称":"测试公司2","联系人":"李四","职位":"总监","电话":"13900139000","行ID":"456"}]
    请检查数据格式是否正确。"""

    test_cases = [
        ("标准JSON数组", test_case1),
        ("Unicode转义的JSON数组", test_case2),
        ("嵌入在文本中的JSON数组", test_case3),
    ]

    for case_name, content in test_cases:
        print(f"\n测试案例: {case_name}")
        try:
            # 清理内容
            content = content.strip()

            # 测试方案1: 直接解析JSON (如果内容是JSON数组)
            if content.startswith("[") and content.endswith("]"):
                try:
                    print("尝试直接解析...")
                    results = json.loads(content)
                    if isinstance(results, list):
                        print(f"✅ 直接解析成功，得到 {len(results)} 个项目")
                        if len(results) > 0:
                            print(f"  第一项: {results[0]}")
                    else:
                        print(f"❌ 解析结果不是列表: {type(results)}")
                except Exception as e:
                    print(f"❌ 直接解析失败: {e}")

                    # 测试方案2: 尝试解码Unicode转义字符后解析
                    try:
                        print("尝试解码Unicode转义字符后解析...")
                        cleaned_content = content.encode("utf-8").decode(
                            "unicode_escape"
                        )
                        results = json.loads(cleaned_content)
                        if isinstance(results, list):
                            print(
                                f"✅ Unicode解码后解析成功，得到 {len(results)} 个项目"
                            )
                            if len(results) > 0:
                                print(f"  第一项: {results[0]}")
                        else:
                            print(f"❌ 解析结果不是列表: {type(results)}")
                    except Exception as e2:
                        print(f"❌ Unicode解码后解析仍失败: {e2}")

            # 测试方案3: 查找JSON数组部分
            else:
                print("内容不是JSON数组格式，尝试提取JSON数组部分...")
                try:
                    start_idx = content.find("[")
                    end_idx = content.rfind("]")
                    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                        print("❌ 无法找到有效的JSON数组范围")
                    else:
                        json_part = content[start_idx : end_idx + 1]
                        results = json.loads(json_part)
                        if isinstance(results, list):
                            print(f"✅ 提取并解析成功，得到 {len(results)} 个项目")
                            if len(results) > 0:
                                print(f"  第一项: {results[0]}")
                        else:
                            print(f"❌ 提取的内容不是有效的JSON数组: {type(results)}")
                except Exception as e:
                    print(f"❌ 尝试提取JSON数组部分失败: {e}")

        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            print(traceback.format_exc())

    print("\n测试完成")


if __name__ == "__main__":
    test_json_parsing()
