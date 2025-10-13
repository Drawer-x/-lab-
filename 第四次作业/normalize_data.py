import pymysql
import pandas as pd

# ========== 1️⃣ 数据库连接配置 ==========
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",   # ← 改成你自己的
    "database": "university_analysis",  # ← 目标规范化数据库
    "charset": "utf8mb4"
}

# ========== 2️⃣ 连接数据库 ==========
try:
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ 成功连接数据库！")
except Exception as e:
    print("❌ 数据库连接失败:", e)
    exit()

# ========== 3️⃣ 读取原始表数据 ==========
try:
    df = pd.read_sql("SELECT * FROM university_data.disciplines", conn)
    print(f"📦 从原始表读取 {len(df)} 条记录")
except Exception as e:
    print("❌ 无法读取原始表数据，请确认数据库和表是否存在:", e)
    exit()

# ========== 4️⃣ 去重提取学科和学校 ==========
subjects = df['subject'].drop_duplicates().reset_index(drop=True)
institutions = df['institution'].drop_duplicates().reset_index(drop=True)

# 创建映射字典
subject_map = {}
institution_map = {}

print("🧩 正在创建 subjects 和 institutions 表...")

# 插入学科数据
cursor.execute("DELETE FROM subjects;")
for subj in subjects:
    cursor.execute("INSERT INTO subjects (subject_name) VALUES (%s);", (subj,))
    subject_map[subj] = cursor.lastrowid

# 插入学校数据
cursor.execute("DELETE FROM institutions;")
for inst in institutions:
    # 尝试自动识别国家（简单规则：取 institution 中的最后部分）
    if "CHINA" in inst.upper():
        country = "CHINA MAINLAND"
        region = "ASIA"
    elif "USA" in inst.upper() or "UNITED STATES" in inst.upper():
        country = "USA"
        region = "NORTH AMERICA"
    elif "UK" in inst.upper() or "ENGLAND" in inst.upper():
        country = "UNITED KINGDOM"
        region = "EUROPE"
    elif "FRANCE" in inst.upper():
        country = "FRANCE"
        region = "EUROPE"
    elif "NETHERLANDS" in inst.upper():
        country = "NETHERLANDS"
        region = "EUROPE"
    else:
        country = None
        region = None

    cursor.execute(
        "INSERT INTO institutions (institution_name, country, region) VALUES (%s, %s, %s);",
        (inst, country, region)
    )
    institution_map[inst] = cursor.lastrowid

conn.commit()
print(f"✅ 已创建 {len(subjects)} 个学科，{len(institutions)} 个机构")

# ========== 5️⃣ 插入 metrics 数据 ==========
print("📊 正在插入 metrics 数据（主表）...")
cursor.execute("DELETE FROM metrics;")

success, fail = 0, 0
for _, row in df.iterrows():
    try:
        subject_id = subject_map[row['subject']]
        institution_id = institution_map[row['institution']]

        sql = """
            INSERT INTO metrics (subject_id, institution_id, `rank`, wos_documents, cites, cites_per_paper, top_papers)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            subject_id,
            institution_id,
            int(row['rank']) if not pd.isna(row['rank']) else None,
            int(row['wos_documents']) if not pd.isna(row['wos_documents']) else None,
            int(row['cites']) if not pd.isna(row['cites']) else None,
            float(row['cites_per_paper']) if not pd.isna(row['cites_per_paper']) else None,
            int(row['top_papers']) if not pd.isna(row['top_papers']) else None
        ))
        success += 1
    except Exception as e:
        fail += 1
        print(f"⚠️ 插入失败：{e}")

conn.commit()
print(f"✅ 数据迁移完成：成功 {success} 条，失败 {fail} 条")

# ========== 6️⃣ 检查结果 ==========
cursor.execute("SELECT COUNT(*) FROM subjects;")
print(f"📘 subjects 表共 {cursor.fetchone()[0]} 条")

cursor.execute("SELECT COUNT(*) FROM institutions;")
print(f"🏫 institutions 表共 {cursor.fetchone()[0]} 条")

cursor.execute("SELECT COUNT(*) FROM metrics;")
print(f"📊 metrics 表共 {cursor.fetchone()[0]} 条")

cursor.close()
conn.close()
print("🎉 规范化迁移完成！")
