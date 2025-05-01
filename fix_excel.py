#!/usr/bin/env python3
f = open("app.py", "r")
content = f.read()
f.close()
old_code = """                final_df.to_excel(
                    final_output_path, index=False, engine=\"openpyxl\", na_rep=\"\"
                )"""
new_code = """                # 使用多Sheet页Excel保存最终结果文件
                postprocessor.create_multi_sheet_excel(
                    final_df, final_output_path, config
                )"""
content = content.replace(old_code, new_code)
f = open("app.py", "w")
f.write(content)
f.close()
print("修改成功")
