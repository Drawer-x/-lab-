import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========== 1. 读取清洗后的数据 ==========
df = pd.read_csv(r"C:\Users\xze97\Desktop\Programming\python\lab\第六次作业\esi_all_clean.csv")

# ========== 2. 聚合到“学校”层面 ==========
# 对每所大学统计整体科研特征
agg_df = df.groupby("university").agg(
    total_papers=("papers", "sum"),
    total_cites=("cites", "sum"),
    avg_cites_per_paper=("cites_per_paper", "mean"),
    total_top_papers=("top_papers", "sum"),
    best_rank=("rank", "min"),
    avg_rank=("rank", "mean")
).reset_index()

print(f"✅ 已聚合 {len(agg_df)} 所学校的数据")

# ========== 3. 数据标准化 ==========
features = ["total_papers", "total_cites", "avg_cites_per_paper", "total_top_papers", "best_rank", "avg_rank"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(agg_df[features])

# ========== 4. 聚类 ==========
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
agg_df["cluster_id"] = kmeans.fit_predict(X_scaled)

# ========== 5. 查找华东师范大学所在簇 ==========
# 注意：文件中可能学校名是大写或含地区后缀
# 我们先统一成大写后匹配
agg_df["university_upper"] = agg_df["university"].str.upper()

target_name = "EAST CHINA NORMAL UNIVERSITY"
target_upper = target_name.upper()

if target_upper not in agg_df["university_upper"].values:
    print("⚠️ 未找到 East China Normal University，请检查名称是否匹配。")
else:
    cluster_id = agg_df.loc[agg_df["university_upper"] == target_upper, "cluster_id"].iloc[0]
    similar_schools = agg_df[agg_df["cluster_id"] == cluster_id].sort_values("avg_rank")
    
    print(f"\n✅ 华东师范大学所在的簇编号: {cluster_id}")
    print("以下学校与华东师范大学处于同一类（科研画像相似）：\n")
    print(similar_schools[["university", "total_papers", "total_cites", "avg_rank"]].head(20))

# ========== 6. 结果保存 ==========
output_path = r"C:\Users\xze97\Desktop\Programming\python\lab\第六次作业\similar_schools.csv"
similar_schools.to_csv(output_path, index=False)
print(f"\n✅ 已保存相似学校结果: {output_path}")
