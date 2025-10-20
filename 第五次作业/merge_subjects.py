import pandas as pd
import os

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_data = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        subject = file.replace(".csv", "")
        path = os.path.join(DATA_DIR, file)
        try:
            # ✅ 跳过第一行，用第二行作为表头
            df = pd.read_csv(path, encoding="latin1", header=1)
            
            # 只保留必要列并重命名
            rename_map = {
                "Institutions": "univ",
                "Countries/Regions": "country",
                "Web of Science Documents": "docs",
                "Cites": "cites",
                "Cites/Paper": "cpp",
                "Top Papers": "top_papers"
            }
            df = df.rename(columns=rename_map)
            
            # 过滤缺失数据
            df = df[["univ","country","docs","cites","cpp","top_papers"]]
            df["subject"] = subject
            
            all_data.append(df)
            print(f"✅ 已读取 {file} ({len(df)} 行)")
        except Exception as e:
            print(f"❌ 读取 {file} 出错: {e}")

# 合并
merged = pd.concat(all_data, ignore_index=True)
merged.to_csv(os.path.join(OUTPUT_DIR, "esi_all.csv"), index=False, encoding="utf-8-sig")

print(f"\n✅ 合并完成，共 {len(merged)} 行，输出文件：outputs/esi_all.csv")
