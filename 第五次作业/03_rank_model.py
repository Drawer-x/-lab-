import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
from math import sqrt
import os

# ==============================
# 1. 路径设置
# ==============================
INPUT = "outputs/esi_all.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 2. 安全读取 CSV（自动尝试编码）
# ==============================
try:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
except Exception:
    df = pd.read_csv(INPUT, encoding="latin1")

# ==============================
# 3. 标准化列名
# ==============================
def normalize_col(col):
    col = str(col).replace("\ufeff", "").strip().lower()
    col = " ".join(col.split())
    return col

df.columns = [normalize_col(c) for c in df.columns]

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

# ==============================
# 4. 转换数值类型并清理数据
# ==============================
for col in ["docs","cites","cpp","top_papers"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["docs","cites","cpp","top_papers"])
df = df[df["docs"] > 0]
df = df.dropna(subset=["subject"])

# ==============================
# 5. 构造排名与百分位
# ==============================
df["rank"] = df.groupby("subject")["docs"].rank(ascending=False)
df["rank_percentile"] = df.groupby("subject")["rank"].rank(pct=True)

before = len(df)
df = df.dropna(subset=["rank_percentile"])
after = len(df)
print(f"清理后有效样本数：{after}（已删除 {before-after} 行缺失排名数据）")

# ==============================
# 6. 特征工程
# ==============================
df["log_docs"] = np.log1p(df["docs"])
df["log_cites"] = np.log1p(df["cites"])
df["top_ratio"] = df["top_papers"] / df["docs"].replace(0, np.nan)
df["top_ratio"] = df["top_ratio"].fillna(0)

features = ["log_docs","log_cites","cpp","top_papers","top_ratio"]
X = df[features]
y = df["rank_percentile"]

# 再次确保无 NaN
mask = ~y.isna()
X, y = X[mask], y[mask]
print(f"最终训练样本：{len(X)} 行")

# ==============================
# 7. 数据集划分
# ==============================
train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.4, random_state=42)
valid_X, test_X, valid_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)

# ==============================
# 8. 训练随机森林模型
# ==============================
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(train_X, train_y)

# ==============================
# 9. 验证集评估
# ==============================
pred_v = rf.predict(valid_X)
mae_v = mean_absolute_error(valid_y, pred_v)
rmse_v = sqrt(mean_squared_error(valid_y, pred_v))  # 兼容旧版 sklearn
spr_v = spearmanr(valid_y, pred_v).correlation
print(f"验证集  MAE={mae_v:.4f} RMSE={rmse_v:.4f} Spearman={spr_v:.3f}")

# ==============================
# 10. 测试集评估
# ==============================
pred_t = rf.predict(test_X)
mae_t = mean_absolute_error(test_y, pred_t)
rmse_t = sqrt(mean_squared_error(test_y, pred_t))  # 兼容旧版 sklearn
spr_t = spearmanr(test_y, pred_t).correlation
print(f"测试集  MAE={mae_t:.4f} RMSE={rmse_t:.4f} Spearman={spr_t:.3f}")

# ==============================
# 11. 特征重要性输出
# ==============================
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\n特征重要性：")
print(imp)

# ==============================
# 12. 输出预测结果
# ==============================
df_test = df.loc[test_X.index, ["univ","subject"]].copy()
df_test["rank_percentile"] = test_y
df_test["pred_percentile"] = pred_t
df_test.to_csv(os.path.join(OUTPUT_DIR, "rank_predictions_test.csv"), index=False, encoding="utf-8-sig")
print("\n✅ 已保存：outputs/rank_predictions_test.csv")
