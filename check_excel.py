import pandas as pd
import sys


def check_excel_sheets(file_path):
    print(f"检查Excel文件: {file_path}")

    # 读取所有Sheet页的名称
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    print(f"文件包含 {len(sheet_names)} 个Sheet页: {sheet_names}")

    # 检查每个Sheet页的内容
    for sheet in sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet)
            print(f"Sheet '{sheet}': {len(df)} 行, {len(df.columns)} 列")
            print(f"  列: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"  前1行: {df.iloc[0].to_dict()}")
            else:
                print("  (空Sheet页)")
        except Exception as e:
            print(f"  错误: {str(e)}")

    print("检查完成")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "outputs/task_20250509_144112/final_task_20250509_144112.xlsx"

    check_excel_sheets(file_path)
