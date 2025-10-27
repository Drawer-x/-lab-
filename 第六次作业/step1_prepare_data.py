import pandas as pd

# ========== 1. 读取原始 ESI 数据 ==========
# 使用绝对路径，确保不会再出现找不到文件的问题
df = pd.read_csv(r"C:\Users\xze97\Desktop\Programming\python\lab\第六次作业\esi_all.csv")

# ========== 2. 统一列名 ==========
df = df.rename(columns={
    "univ": "university",          # 学校名称
    "docs": "papers",              # 论文数量
    "cpp": "cites_per_paper",      # 每篇平均被引
    "top_paper": "top_papers"      # 高被引论文数
})

# ========== 3. 为每个学科生成排名列 ==========
# 每个 subject 内从 1 开始编号作为 rank
df["rank"] = df.groupby("subject").cumcount() + 1

# ========== 4. 打印前10行预览 ==========
print("✅ 数据预览（前10行）：")
print(df.head(10))

# ========== 5. 保存清洗后的文件 ==========
output_path = r"C:\Users\xze97\Desktop\Programming\python\lab\第六次作业\esi_all_clean.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ 已生成清洗文件: {output_path}")
