import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# === Step 1: 数据读取与处理 ===
df = pd.read_csv("ecoli_TE_CDS_final.csv")
cds_list = df["CDS_sequence"].astype(str).str.upper().str.replace("U", "T").str.replace("N", "A")
mrl_list = df["TE"].values.astype(float)

def extract_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq), 3) if len(seq[i:i+3]) == 3 and set(seq[i:i+3]).issubset("ATCG")]

codon_seqs = [extract_codons(seq) for seq in cds_list]
filtered = [(seq, mrl) for seq, mrl in zip(codon_seqs, mrl_list) if len(seq) > 0]
codon_seqs = [f[0] for f in filtered]
y = [f[1] for f in filtered]

all_codons = sorted(set(c for seq in codon_seqs for c in seq))
encoder = LabelEncoder()
encoder.fit(all_codons)
PAD_IDX = len(encoder.classes_)
MAX_LEN = max(len(seq) for seq in codon_seqs)

print(f"PAD_IDX: {PAD_IDX}")
print(f"MAX_LEN: {MAX_LEN}")
exit()
X_encoded = [encoder.transform(seq).tolist() + [PAD_IDX] * (MAX_LEN - len(seq)) for seq in codon_seqs]
X_tensor = torch.tensor(X_encoded, dtype=torch.long)
scaler = StandardScaler()
y_tensor = torch.tensor(scaler.fit_transform(np.array(y).reshape(-1, 1)), dtype=torch.float32)

# === Step 2: 数据划分与封装 ===
X_temp, X_test, y_temp, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

class CodonDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.squeeze(1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(CodonDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(CodonDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(CodonDataset(X_test, y_test), batch_size=64)

# === Step 3: Transformer 模型定义 ===
class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD_IDX)
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_LEN, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1)]
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(1)

# === Step 4: 模型训练与评估 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerRegressor(len(encoder.classes_)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val_loss = float("inf")
train_losses, val_losses = [], []

for epoch in range(30):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * len(yb)
    val_loss /= len(val_loader.dataset)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_transformer_model.pth")

# === 可视化 ===
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Transformer: Train vs Val Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_transformer.png")

# === 测试评估 ===
model.load_state_dict(torch.load("best_transformer_model.pth"))
model.eval()
preds, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy()
        preds.extend(out)
        targets.extend(yb.numpy())

y_pred = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
y_true = scaler.inverse_transform(np.array(targets).reshape(-1, 1)).flatten()

r2 = r2_score(y_true, y_pred)
pearson_corr, p_val = pearsonr(y_true, y_pred)
print(f'R^2: {r2:.4f}, Pearson: {pearson_corr:.4f}, p={p_val:.2g}')

# === 保存 Pearson 散点图 ===
plt.figure(figsize=(6, 5))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("Actual Translation Efficiency")
plt.ylabel("Predicted Translation Efficiency")
plt.title("Actual vs Predicted (Transformer)")
plt.grid(True)
plt.text(0.05, 0.95,
         f"Pearson r = {pearson_corr:.4f}\np = {p_val:.2g}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         fontsize=12,
         bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.6))
plt.tight_layout()
plt.savefig("pearson_scatter_transformer.png")
plt.close()

# === 保存预测结果 CSV ===
df_result = pd.DataFrame({
    "Actual": y_true,
    "Predicted": y_pred
})
df_result.to_csv("pred_vs_actual_transformer.csv", index=False)
print("📦 保存完毕：best_transformer_model.pth、loss_curve_transformer.png、pearson_scatter_transformer.png、pred_vs_actual_transformer.csv")


