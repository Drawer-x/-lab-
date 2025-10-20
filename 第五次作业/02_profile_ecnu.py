import pandas as pd
import numpy as np
import os

INPUT = "outputs/esi_all.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) 安全读取（自动尝试 utf-8 / latin1）
try:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
except Exception:
    df = pd.read_csv(INPUT, encoding="latin1")

# 2) 统一列名格式
def normalize_col(col):
    col = str(col).replace("\ufeff", "").strip().lower()
    col = " ".join(col.split())
    return col

df.columns = [normalize_col(c) for c in df.columns]

# 3) 重命名映射（与前面的 merge、cluster 保持一致）
rename_map = {}
for c in df.columns:
    lc = c
    if "institution" in lc:
        rename_map[c] = "univ"
    elif "countr" in lc or "region" in lc:
        rename_map[c] = "country"
    elif "document" in lc or "web of sc" in lc:
        rename_map[c] = "docs"
    elif "cites/paper" in lc or ("cite" in lc and "paper" in lc):
        rename_map[c] = "cpp"
    elif "top" in lc and "paper" in lc:
        rename_map[c] = "top_papers"
    elif lc == "cites":
        rename_map[c] = "cites"
    elif lc == "subject":
        rename_map[c] = "subject"

df = df.rename(columns=rename_map)
print("标准化后列名：", df.columns.tolist())

# 4) 检查关键列
required = ["univ","country","docs","cites","cpp","top_papers","subject"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"缺少关键列：{missing}")

# 5) 转数值
for col in ["docs","cites","cpp","top_papers"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["univ","subject"])

# 6) 设定目标学校
ECNU = "EAST CHINA NORMAL UNIVERSITY"
SUBJECT_COLS = ["docs","cites","cpp","top_papers"]

ecnu = df[df["univ"].str.strip().str.upper() == ECNU].copy()
if ecnu.empty:
    raise ValueError("⚠️ 数据中未找到 EAST CHINA NORMAL UNIVERSITY，请确认名称是否一致")

# 7) 全球与中国统计
world = df.groupby("subject")[SUBJECT_COLS].agg(["mean","std"])
china = df[df["country"].str.contains("CHINA", case=False, na=False)] \
    .groupby("subject")[SUBJECT_COLS].agg(["median"])

# 8) 计算 z-score / RSI
def zscore(row, base):
    subj = row["subject"]
    out = {}
    for col in SUBJECT_COLS:
        mu = base.loc[subj, (col,"mean")]
        sd = base.loc[subj, (col,"std")] or 1
        out[col+"_z_world"] = (row[col]-mu)/sd
    return pd.Series(out)

def rsi(row, base):
    subj = row["subject"]
    out = {}
    for col in SUBJECT_COLS:
        med = base.loc[subj, (col,"median")]
        q75 = df[df["subject"]==subj][col].quantile(0.75)
        q25 = df[df["subject"]==subj][col].quantile(0.25)
        iqr = max(q75-q25,1e-6)
        out[col+"_rsi_cn"] = (row[col]-med)/iqr
    return pd.Series(out)

prof = ecnu.join(ecnu.apply(lambda r: zscore(r, world), axis=1))
prof = prof.join(ecnu.apply(lambda r: rsi(r, china), axis=1))

score_cols = [c for c in prof.columns if c.endswith("_z_world") or c.endswith("_rsi_cn")]
prof["strength_score"] = prof[score_cols].mean(axis=1)

top_strength = prof.sort_values("strength_score", ascending=False).head(8)
top_weak = prof.sort_values("strength_score", ascending=True).head(8)

# 9) 输出结果
prof.to_csv(os.path.join(OUTPUT_DIR, "ecnu_profile_by_subject.csv"), index=False, encoding="utf-8-sig")
top_strength.to_csv(os.path.join(OUTPUT_DIR, "ecnu_strength_subjects.csv"), index=False, encoding="utf-8-sig")
top_weak.to_csv(os.path.join(OUTPUT_DIR, "ecnu_weak_subjects.csv"), index=False, encoding="utf-8-sig")

print("✅ 已保存：ECNU 学科画像、优势与短板结果")
