import os
import pandas as pd

# 1. 文件夹路径
folder_path = r"C:\Users\xze97\Desktop\Programming\python\lab\第三次作业\esi_results"

# 2. 匹配关键字
keyword = "east china normal university"

# 3. 存放结果
all_results = []

# 4. 遍历所有 Excel 文件
for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(folder_path, file)
        print(f"📄 正在处理：{file}")

        try:
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # 只取第二列作为“学校名称”列
                if df.shape[1] < 2:
                    continue  # 如果列数不够就跳过

                school_col = df.columns[1]  # 第二列
                mask = df[school_col].astype(str).str.lower().str.contains(keyword)
                ecnu_df = df[mask]

                if not ecnu_df.empty:
                    ecnu_df["来源文件"] = file
                    ecnu_df["Sheet"] = sheet_name
                    all_results.append(ecnu_df)

        except Exception as e:
            print(f"❌ 处理 {file} 出错：{e}")

# 5. 合并结果 & 导出
if all_results:
    result_df = pd.concat(all_results, ignore_index=True)
    save_path = os.path.join(folder_path, "#ECNU_all.xlsx")
    result_df.to_excel(save_path, index=False)
    print(f"\n✅ 已提取所有 East China Normal University 数据，保存到：\n{save_path}")
else:
    print("\n⚠️ 没有找到任何匹配的数据。")
