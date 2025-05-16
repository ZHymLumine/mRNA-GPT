import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import joblib

# === 步骤1: 数据读取与处理 ===
print("正在读取数据...")
df = pd.read_csv("ecoli_TE_CDS_final.csv")
cds_list = df["CDS_sequence"].astype(str).str.upper().str.replace("U", "T").str.replace("N", "A")
mrl_list = df["TE"].values.astype(float)

def extract_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq), 3) if len(seq[i:i+3]) == 3 and set(seq[i:i+3]).issubset("ATCG")]

# 提取密码子并过滤无效序列
codon_seqs = [extract_codons(seq) for seq in cds_list]
filtered = [(seq, mrl) for seq, mrl in zip(codon_seqs, mrl_list) if len(seq) > 0]
codon_seqs = [f[0] for f in filtered]
y = [f[1] for f in filtered]

# 计算每个序列的密码子频率特征
all_codons = sorted(set(c for seq in codon_seqs for c in seq))
print(f"总共有 {len(all_codons)} 种密码子")

# 为每个序列创建密码子频率特征
X_features = []
for seq in codon_seqs:
    codon_count = {codon: 0 for codon in all_codons}
    for codon in seq:
        codon_count[codon] += 1
    
    # 计算频率而不是计数
    total_codons = len(seq)
    codon_freq = {codon: count/total_codons for codon, count in codon_count.items()}
    
    # 添加其他可能有用的特征
    features = list(codon_freq.values())
    features.append(len(seq))  # 添加序列长度作为特征
    
    X_features.append(features)

X = np.array(X_features)
print(f"特征矩阵形状: {X.shape}")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_array = np.array(y)

# === 步骤2: 数据划分 ===
print("划分数据集...")
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_array, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# === 步骤3: 随机森林模型训练 ===
print("训练随机森林模型...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# === 步骤4: 模型评估 ===
print("评估模型性能...")
# 验证集评估
y_val_pred = model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
val_pearson, val_p = pearsonr(y_val, y_val_pred)
print(f"验证集: R^2 = {val_r2:.4f}, Pearson r = {val_pearson:.4f}, p = {val_p:.2g}")

# 测试集评估
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_pearson, test_p = pearsonr(y_test, y_test_pred)
print(f"测试集: R^2 = {test_r2:.4f}, Pearson r = {test_pearson:.4f}, p = {test_p:.2g}")

# === 步骤5: 特征重要性分析 ===
feature_names = all_codons + ["序列长度"]
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("随机森林模型的特征重要性")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(min(20, X.shape[1])), [feature_names[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig("rf_feature_importance.png")

# === 步骤6: 保存预测结果和模型 ===
# 保存预测散点图
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("实际翻译效率")
plt.ylabel("预测翻译效率")
plt.title("实际值 vs 预测值 (随机森林)")
plt.grid(True)
plt.text(0.05, 0.95,
         f"Pearson r = {test_pearson:.4f}\np = {test_p:.2g}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         fontsize=12,
         bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.6))
plt.tight_layout()
plt.savefig("pearson_scatter_randomforest.png")

# 保存预测结果
df_result = pd.DataFrame({
    "实际值": y_test,
    "预测值": y_test_pred
})
df_result.to_csv("pred_vs_actual_randomforest.csv", index=False)

# 保存模型
joblib.dump(model, "best_randomforest_model.joblib")
joblib.dump(scaler, "feature_scaler.joblib")
joblib.dump(feature_names, "feature_names.joblib")

print("📦 保存完毕：best_randomforest_model.joblib、feature_scaler.joblib、feature_names.joblib、rf_feature_importance.png、pearson_scatter_randomforest.png、pred_vs_actual_randomforest.csv") 