import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

INPUT = "outputs/esi_all.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) 读取（utf-8-sig 优先，失败回退 latin1）
try:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
except Exception:
    df = pd.read_csv(INPUT, encoding="latin1")

# 2) 规范化列名：去空格、去BOM、小写
def normalize_col(col: str) -> str:
    col = str(col).replace("\ufeff", "").strip().lower()
    col = " ".join(col.split())  # 折叠多空格
    return col

df.columns = [normalize_col(c) for c in df.columns]

# 3) 统一映射为目标列名
rename_map = {}
for c in df.columns:
    lc = c
    if ("institution" in lc) or ("institutions" in lc):
        rename_map[c] = "univ"
    elif ("countr" in lc) or ("region" in lc):
        rename_map[c] = "country"
    elif ("web of sc" in lc) or ("document" in lc) or ("docs" in lc):
        rename_map[c] = "docs"
    elif (("cites/paper" in lc) or ("cites per paper" in lc) or (("cite" in lc) and ("paper" in lc))):
        rename_map[c] = "cpp"
    elif ("top" in lc) and ("paper" in lc):
        rename_map[c] = "top_papers"
    elif (lc == "cites") or (lc.endswith(" cites")) or (" cites" in lc):
        rename_map[c] = "cites"
    elif lc == "subject":
        rename_map[c] = "subject"

df = df.rename(columns=rename_map)

# 4) 强校验：必须列
required = ["univ", "country", "docs", "cites", "cpp", "top_papers", "subject"]
missing = [c for c in required if c not in df.columns]
print("标准化后列名：", df.columns.tolist())
if missing:
    raise ValueError(f"缺少关键列：{missing}。请确认 merge_subjects.py 已按 header=1 合并并输出 esi_all.csv。")

# 5) 去掉空白学校名、修整字符串
df["univ"] = df["univ"].astype(str).str.strip()
df["country"] = df["country"].astype(str).str.strip()
df["subject"] = df["subject"].astype(str).str.strip()
df = df[df["univ"].ne("") & df["univ"].notna()]

# 6) 数值列安全转换
for col in ["docs", "cites", "cpp", "top_papers"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["docs","cites","cpp","top_papers"])

# 7) 聚合到学校层面（加权平均 cpp）
def weighted_mean(x, w):
    wsum = np.maximum(w.sum(), 1)
    return (x * w).sum() / wsum

g = df.groupby("univ", as_index=False).apply(
    lambda g: pd.Series({
        "docs": g["docs"].sum(),
        "cites": g["cites"].sum(),
        "top_papers": g["top_papers"].sum(),
        "cpp": weighted_mean(g["cpp"], g["docs"]),
        "country": g["country"].mode().iat[0] if not g["country"].mode().empty else np.nan
    })
).reset_index(drop=True)

# 8) 标准化 + 选 K
X = g[["docs","cites","cpp","top_papers"]].fillna(0.0).values
Xs = StandardScaler().fit_transform(X)

best_k, best_score, best_model = None, -1, None
for k in range(3, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Xs)
    score = silhouette_score(Xs, labels)
    if score > best_score:
        best_k, best_score, best_model = k, score, km

print(f"最佳聚类数: {best_k}  (轮廓系数={best_score:.3f})")
g["cluster"] = best_model.predict(Xs)
g["tier"] = g["cluster"].map(lambda x: f"Type-{x}")

# 9) 计算与 ECNU 相似度
target = "EAST CHINA NORMAL UNIVERSITY"
if target in g["univ"].values:
    sim = cosine_similarity(Xs, Xs)
    idx = g.index[g["univ"] == target][0]
    g["sim_to_ECNU"] = sim[idx]
else:
    g["sim_to_ECNU"] = np.nan
    print("⚠️ 未在合并数据中找到：EAST CHINA NORMAL UNIVERSITY。")

cluster_profile = g.groupby("tier")[["docs","cites","cpp","top_papers"]].mean().round(2)

# 10) 保存
g.to_csv(os.path.join(OUTPUT_DIR, "univ_clusters.csv"), index=False, encoding="utf-8-sig")
cluster_profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profile.csv"), encoding="utf-8-sig")
print("✅ 已保存：outputs/univ_clusters.csv, outputs/cluster_profile.csv")
