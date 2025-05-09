import os
import time
import json

# 测试时间戳格式的任务ID
timestamp = time.strftime("%Y%m%d_%H%M%S")
task_id = f"task_{timestamp}"
print(f"生成的时间戳任务ID: {task_id}")

# 创建测试目录
test_dir = f"uploads/{task_id}"
os.makedirs(test_dir, exist_ok=True)
print(f"创建测试目录: {test_dir}")

# 创建测试映射文件
test_mapping = {
    "batch_mapping": {
        "file0_batch_0_100": ["id1", "id2", "id3"],
        "file0_batch_1_100": ["id4", "id5", "id6"],
    },
    "files": ["test_file1.xlsx", "test_file2.xlsx"],
    "total_rows": 6,
    "batch_size": 100,
}

test_mapping_path = os.path.join(test_dir, "batch_mapping.json")
with open(test_mapping_path, "w", encoding="utf-8") as f:
    json.dump(test_mapping, f, ensure_ascii=False, indent=2)
print(f"创建测试映射文件: {test_mapping_path}")

# 创建测试结果文件
test_result = {
    "id": "test-id",
    "custom_id": "file0_batch_0_100",
    "response": {
        "status_code": 200,
        "body": {
            "choices": [
                {
                    "message": {
                        "content": '[{"企业名称":"测试公司","联系人":"张三","职位":"经理","电话":"12345678901","行ID":"id1"}]'
                    }
                }
            ]
        },
    },
}

test_result_path = os.path.join(test_dir, "batch_result.jsonl")
with open(test_result_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(test_result, ensure_ascii=False))
print(f"创建测试结果文件: {test_result_path}")

print("测试完成!")
