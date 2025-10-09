import os
import pandas as pd

# 1. 你的文件夹路径
folder_path = r"C:\Users\xze97\Desktop\Programming\python\lab\第三次作业\esi_results"

# 2. 匹配关键词
keyword = "EAST CHINA NORMAL UNIVERSITY"

# 3. 存放所有结果的列表
results = []

# 4. 遍历文件夹中的所有 Excel 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(folder_path, filename)
        print(f"正在处理：{filename}")

        try:
            # 读取第一个sheet
            df = pd.read_excel(file_path, header=0)

            # 第二列是学校名称，匹配关键词
            matched = df[df.iloc[:, 1].astype(str).str.strip().str.upper() == keyword]

            if not matched.empty:
                row = matched.iloc[0]  # 假设每个文件只有一行匹配

                results.append({
                    "学科": os.path.splitext(filename)[0],
                    "排名": row.iloc[0],   # 第1列
                    "Web of Science Documents": row.iloc[3],  # 第4列
                    "Cites": row.iloc[4],  # 第5列
                    "Cites/Paper": row.iloc[5],  # 第6列
                    "Top Papers": row.iloc[6]   # 第7列
                })

        except Exception as e:
            print(f"⚠ 处理 {filename} 出错：{e}")

# 5. 生成汇总Excel
if results:
    result_df = pd.DataFrame(results)
    output_file = os.path.join(folder_path, "ECNU_学科数据汇总.xlsx")
    result_df.to_excel(output_file, index=False)
    print(f"\n✅ 提取完成，结果已保存至：{output_file}")
else:
    print("\n❌ 没有在任何文件中找到 ECNU 的记录。")
