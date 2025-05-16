import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ====== 1. 数据读取与处理 ======
df = pd.read_csv("E.Coli_proteins.csv")
cds_list = df["Sequence"].astype(str).str.upper().str.replace("U", "T").str.replace("N", "A")
y_list = df["Value"].astype(int).values
split = df["Split"].values

# 拆codon
def extract_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq), 3) if len(seq[i:i+3]) == 3 and set(seq[i:i+3]).issubset("ATCG")]

codon_seqs = [extract_codons(seq) for seq in cds_list]
filtered = [(codons, label, sp) for codons, label, sp in zip(codon_seqs, y_list, split) if len(codons) > 0]
codon_seqs = [item[0] for item in filtered]
y = np.array([item[1] for item in filtered])
split = np.array([item[2] for item in filtered])

# 编码codon
all_codons = list(set(c for seq in codon_seqs for c in seq))
encoder = LabelEncoder()
encoder.fit(all_codons)
X_encoded = [encoder.transform(seq) for seq in codon_seqs]

# padding
max_len = max(len(seq) for seq in X_encoded)
X_padded = np.array([np.pad(seq, (0, max_len - len(seq))) for seq in X_encoded])

# 划分训练/验证/测试集
X_train = X_padded[split == "train"]
y_train = y[split == "train"]
X_val = X_padded[split == "val"]
y_val = y[split == "val"]
X_test = X_padded[split == "test"]
y_test = y[split == "test"]

# ====== 2. 数据集包装 ======
class CodonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_loader = DataLoader(CodonDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
val_loader = DataLoader(CodonDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(CodonDataset(X_test, y_test), batch_size=batch_size)

# ====== 3. 定义简化并防过拟合的Transformer分类器 ======
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.register_buffer("positional_encoding", self._generate_sinusoidal_positional_encoding(max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*2, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def _generate_sinusoidal_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ====== 4. 训练设置 ======
model = TransformerClassifier(
    vocab_size=len(encoder.classes_),
    embed_dim=96,
    nhead=4,
    num_layers=2,
    num_classes=3,
    max_len=max_len
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
best_epoch = 0
early_stop_patience = 8
early_stop_counter = 0

# ====== 5. 正式训练 ======
for epoch in range(40):
    model.train()
    train_loss, train_correct = 0, 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        attention_mask = (x_batch == 0)
        optimizer.zero_grad()
        outputs = model(x_batch, attention_mask=attention_mask)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y_batch)
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == y_batch).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            attention_mask = (x_batch == 0)
            outputs = model(x_batch, attention_mask=attention_mask)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * len(y_batch)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == y_batch).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

# ====== 6. 绘制Loss和Accuracy曲线（带最佳点） ======
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axvline(x=best_epoch-1, linestyle='--', color='red', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Validation Acc')
plt.axvline(x=best_epoch-1, linestyle='--', color='red', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_acc_curve_transformer_expression_best_marked.png")

# ====== 7. 测试集评估 ======
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        attention_mask = (x_batch == 0)
        logits = model(x_batch, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_transformer_expression_best.png")
plt.close()
