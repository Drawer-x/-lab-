import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ========== 1. 读取清洗后的数据 ==========
df = pd.read_csv(r"C:\Users\xze97\Desktop\Programming\python\lab\第六次作业\esi_all_clean.csv")

# ========== 2. 选择特征与目标 ==========
feature_cols = ["papers", "cites", "cites_per_paper", "top_papers"]
target_col = "rank"

# ========== 3. 数据清洗 ==========
# 去掉任何含有 NaN 的行
df = df.dropna(subset=feature_cols + [target_col])

# 去掉无效或极端值（例如 0 或负数）
for col in feature_cols:
    df = df[df[col] > 0]

# 确保 rank 为正
df = df[df["rank"] > 0]

print(f"✅ 清洗后数据量: {len(df)} 条记录")

# ========== 4. 提取特征与标签 ==========
X = df[feature_cols].values
y = df[target_col].values

# ========== 5. 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 6. 标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 转 tensor
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ========== 7. 定义神经网络 ==========
class RankRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = RankRegressor(in_dim=X_train_t.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 8. 训练 ==========
EPOCHS = 300
for epoch in range(EPOCHS):
    model.train()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  TrainLoss={loss.item():.4f}")

# ========== 9. 预测与评估 ==========
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_t).numpy().flatten()

# 去掉预测中的 NaN
mask = ~np.isnan(y_pred_test)
y_pred_test = y_pred_test[mask]
y_test = y_test[mask]

mse  = mean_squared_error(y_test, y_pred_test)
mape = mean_absolute_percentage_error(y_test, y_pred_test)

print("\n========== 模型评估结果 ==========")
print(f"MSE  = {mse:.4f}")
print(f"MAPE = {mape:.4f}")
