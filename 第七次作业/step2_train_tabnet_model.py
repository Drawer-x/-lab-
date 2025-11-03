import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import torch

# ========== 1. 读取数据 ==========
df = pd.read_csv(r"C:\Users\xze97\Desktop\Programming\python\lab\第七次作业\esi_all_clean.csv")

# ========== 2. 特征选择与清洗 ==========
feature_cols = ["papers", "cites", "cites_per_paper", "top_papers"]
target_col = "rank"

# 去掉缺失与异常
df = df.dropna(subset=feature_cols + [target_col])
for col in feature_cols:
    df = df[df[col] > 0]
df = df[df[target_col] > 0]

# 对数值量级差异大的列取 log，避免梯度爆炸
for col in ["papers", "cites", "top_papers"]:
    df[col] = np.log1p(df[col])

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# ========== 3. 划分训练集/测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TabNet 通常不需要显式标准化，但我们保持一致
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ========== 4. 定义 TabNet 回归模型 ==========
tabnet_params = {
    "n_d": 16,             # 决策层维度
    "n_a": 16,             # 注意力层维度
    "n_steps": 4,          # 决策步数
    "gamma": 1.5,          # 注意力稀疏度调节
    "n_independent": 2,    # 独立层数
    "n_shared": 2,         # 共享层数
    "lambda_sparse": 1e-4, # 稀疏正则化
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=1e-3),
    "mask_type": "entmax", # 特征选择方式
    "verbose": 10
}

model = TabNetRegressor(**tabnet_params)

# ========== 5. 训练 ==========
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_name=["train", "test"],
    eval_metric=["rmse"],
    max_epochs=200,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# ========== 6. 预测与评估 ==========
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n========== TabNet 模型评估结果 ==========")
print(f"MSE  = {mse:.4f}")
print(f"MAPE = {mape:.4f}")

# ========== 7. 特征重要性分析 ==========
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
importances = model.feature_importances_
plt.bar(feature_cols, importances)
plt.title("Feature Importance (TabNet)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(r"C:\Users\xze97\Desktop\Programming\python\lab\第七次作业\tabnet_feature_importance.png")
plt.show()
