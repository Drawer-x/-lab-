import os
import pandas as pd
import pymysql
import chardet

# ========== 1️⃣ 基本配置 ==========
DATA_DIR = r"C:\Users\xze97\Desktop\Programming\python\lab\第四次作业\data"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",  # 改成你自己的
    "database": "university_data",
    "charset": "utf8mb4"
}

# ========== 2️⃣ 建立数据库连接 ==========
try:
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ 成功连接 MySQL 数据库")
except Exception as e:
    print("❌ 无法连接数据库:", e)
    exit()

# ========== 3️⃣ 列名映射（根据真实文件） ==========
col_map = {
    "Institution": ["Institutions", "Institution", "University"],
    "Web of Science Documents": ["Web of Science Documents"],
    "Cites": ["Cites", "Citations"],
    "Cites/Paper": ["Cites/Paper", "Cites per paper"],
    "Top Papers": ["Top Papers"]
}

total_files = 0
total_rows = 0
skipped_files = []

# ========== 4️⃣ 遍历 CSV 文件 ==========
for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".csv"):
        continue
    total_files += 1
    subject = os.path.splitext(filename)[0]
    filepath = os.path.join(DATA_DIR, filename)

    # 自动检测文件编码
    with open(filepath, 'rb') as f:
        enc = chardet.detect(f.read())['encoding'] or 'utf-8'

    try:
        # 跳过第一行说明性文字
        df = pd.read_csv(filepath, encoding=enc, skiprows=1)
    except Exception as e:
        print(f"⚠️ 无法读取 {filename}: {e}")
        skipped_files.append(filename)
        continue

    # 去除多余空格
    df.columns = [col.strip() for col in df.columns]

    # 处理可能存在的编号列（第一个列名为空或为序号）
    if df.columns[0] == '' or 'Unnamed' in df.columns[0]:
        df.drop(df.columns[0], axis=1, inplace=True)

    # 自动匹配列名
    rename_dict = {}
    for std_col, possible_names in col_map.items():
        for name in df.columns:
            if name.strip() in possible_names:
                rename_dict[name] = std_col
                break
    df.rename(columns=rename_dict, inplace=True)

    # 检查列是否齐全
    required_cols = list(col_map.keys())
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ {filename} 的列名不匹配，跳过")
        skipped_files.append(filename)
        continue

    # 插入数据库
    print(f"📂 正在导入 {filename} ({len(df)} 条记录)...")
    success = 0
    for _, row in df.iterrows():
        try:
            sql = """
                INSERT INTO disciplines (subject, institution, `rank`, wos_documents, cites, cites_per_paper, top_papers)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            # 由于没有 Rank 列，用自动递增序号代替
            cursor.execute(sql, (
                subject,
                row["Institution"],
                int(_ + 1),  # rank = 行号
                int(row["Web of Science Documents"]),
                int(row["Cites"]),
                float(row["Cites/Paper"]),
                int(row["Top Papers"])
            ))
            success += 1
        except Exception as e:
            print(f"❌ 插入失败 ({filename}): {e}")

    conn.commit()
    print(f"✅ {filename} 导入完成 ({success}/{len(df)})")
    total_rows += success

# ========== 5️⃣ 结束处理 ==========
cursor.close()
conn.close()

print("\n🎉 所有文件导入完成！")
print(f"📊 共处理文件数: {total_files}")
print(f"📈 成功导入记录: {total_rows}")
if skipped_files:
    print(f"⚠️ 以下文件被跳过: {', '.join(skipped_files)}")
else:
    print("✅ 所有文件均成功导入")
