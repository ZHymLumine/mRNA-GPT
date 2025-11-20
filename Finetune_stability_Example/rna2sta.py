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


class CodonDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.squeeze(1)

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, pad_idx=64, max_len=1654, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_idx)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        seq_len = x.size(1)
        # 动态调整位置编码
        if seq_len > self.pos_encoding.size(1):
            # 如果序列更长，扩展位置编码
            extended_pos = torch.randn(1, seq_len, self.embed_dim, device=x.device)
            extended_pos[:, :self.pos_encoding.size(1)] = self.pos_encoding
            pos_enc = extended_pos
        else:
            # 使用现有的位置编码
            pos_enc = self.pos_encoding[:, :seq_len]
        
        x = self.embedding(x) + pos_enc
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(1)


class RNAStabilityPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.pad_idx = 64
        self.max_len = 1654
        self.model_path = model_path or "best_transformer_model.pth"
    
    def preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        cds_list = df["Sequence"].astype(str).str.upper().str.replace("U", "T").str.replace("N", "A")
        mrl_list = df["Value"].values.astype(float)
        split_info = df["Split"].values
        
        def extract_codons(seq):
            return [seq[i:i+3] for i in range(0, len(seq), 3) 
                   if len(seq[i:i+3]) == 3 and set(seq[i:i+3]).issubset("ATCG")]

        codon_seqs = [extract_codons(seq) for seq in cds_list]
        filtered = [(seq, mrl, split) for seq, mrl, split in zip(codon_seqs, mrl_list, split_info) if len(seq) > 0]
        codon_seqs = [f[0] for f in filtered]
        y = [f[1] for f in filtered]
        splits = [f[2] for f in filtered]

        all_codons = sorted(set(c for seq in codon_seqs for c in seq))
        self.encoder = LabelEncoder()
        self.encoder.fit(all_codons)
        self.pad_idx = len(self.encoder.classes_)
        self.max_len = max(len(seq) for seq in codon_seqs)

        X_encoded = [self.encoder.transform(seq).tolist() + [self.pad_idx] * (self.max_len - len(seq)) 
                    for seq in codon_seqs]
        X_tensor = torch.tensor(X_encoded, dtype=torch.long)
        self.scaler = StandardScaler()
        y_tensor = torch.tensor(self.scaler.fit_transform(np.array(y).reshape(-1, 1)), dtype=torch.float32)
        
        return X_tensor, y_tensor, splits
    
    def preprocess_data_from_df(self, df):
        """从DataFrame预处理数据（用于获取编码器参数）"""
        cds_list = df["Sequence"].astype(str).str.upper().str.replace("U", "T").str.replace("N", "A")
        mrl_list = df["Value"].values.astype(float)
        split_info = df["Split"].values if "Split" in df.columns else ["train"] * len(df)
        
        def extract_codons(seq):
            return [seq[i:i+3] for i in range(0, len(seq), 3) 
                   if len(seq[i:i+3]) == 3 and set(seq[i:i+3]).issubset("ATCG")]

        codon_seqs = [extract_codons(seq) for seq in cds_list]
        filtered = [(seq, mrl, split) for seq, mrl, split in zip(codon_seqs, mrl_list, split_info) if len(seq) > 0]
        codon_seqs = [f[0] for f in filtered]
        y = [f[1] for f in filtered]
        splits = [f[2] for f in filtered]

        all_codons = sorted(set(c for seq in codon_seqs for c in seq))
        self.encoder = LabelEncoder()
        self.encoder.fit(all_codons)
        self.pad_idx = len(self.encoder.classes_)
        # 不要重新设置max_len，使用现有的或默认值
        if not hasattr(self, 'max_len'):
            self.max_len = max(len(seq) for seq in codon_seqs)

        X_encoded = [self.encoder.transform(seq).tolist() + [self.pad_idx] * (self.max_len - len(seq)) 
                    for seq in codon_seqs]
        X_tensor = torch.tensor(X_encoded, dtype=torch.long)
        self.scaler = StandardScaler()
        y_tensor = torch.tensor(self.scaler.fit_transform(np.array(y).reshape(-1, 1)), dtype=torch.float32)
        
        return X_tensor, y_tensor, splits
    
    def split_data(self, X_tensor, y_tensor, splits):
        """根据数据集中的Split列划分训练、验证和测试数据集"""
        train_mask = [s == 'train' for s in splits]
        val_mask = [s == 'val' for s in splits]
        test_mask = [s == 'test' for s in splits]
        
        X_train = X_tensor[train_mask]
        y_train = y_tensor[train_mask]
        X_val = X_tensor[val_mask]
        y_val = y_tensor[val_mask]
        X_test = X_tensor[test_mask]
        y_test = y_tensor[test_mask]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
        """创建数据加载器"""
        train_loader = DataLoader(CodonDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(CodonDataset(X_val, y_val), batch_size=batch_size)
        test_loader = DataLoader(CodonDataset(X_test, y_test), batch_size=batch_size)
        return train_loader, val_loader, test_loader
    
    def build_model(self, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2):
        """构建Transformer模型"""
        # 使用动态计算的vocab_size，或者默认值
        vocab_size = len(self.encoder.classes_) if hasattr(self, 'encoder') else 64
        self.model = TransformerRegressor(
            vocab_size, 
            self.pad_idx,
            self.max_len,
            embed_dim, 
            num_heads, 
            ff_dim, 
            num_layers
        ).to(self.device)
        return self.model
    
    def train(self, train_loader, val_loader, num_epochs=30, lr=1e-4):
        """训练模型"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_val_loss = float("inf")
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(yb)
            train_loss /= len(train_loader.dataset)

            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item() * len(yb)
            val_loss /= len(val_loader.dataset)
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存模型和相关参数
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                    'encoder': self.encoder,
                    'pad_idx': self.pad_idx,
                    'max_len': self.max_len
                }, self.model_path)

        return train_losses, val_losses
    
    def evaluate(self, test_loader):
        """评估模型"""
        checkpoint = torch.load(self.model_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容旧格式
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        preds, targets = [], []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                out = self.model(xb).cpu().numpy()
                preds.extend(out)
                targets.extend(yb.numpy())

        y_pred = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        y_true = self.scaler.inverse_transform(np.array(targets).reshape(-1, 1)).flatten()

        r2 = r2_score(y_true, y_pred)
        pearson_corr, p_val = pearsonr(y_true, y_pred)
        print(f'R^2: {r2:.4f}, Pearson: {pearson_corr:.4f}, p={p_val:.2g}')

        return y_true, y_pred, r2, pearson_corr, p_val
    
    def plot_results(self, train_losses, val_losses, y_true, y_pred, pearson_corr, p_val):
        """绘制训练结果和评估图表"""
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.title('Transformer: Train vs Val Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("loss_curve_stability_transformer.png")
        plt.close()

        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("Actual mRNA Stability")
        plt.ylabel("Predicted mRNA Stability")
        plt.title("Actual vs Predicted mRNA Stability (Transformer)")
        plt.grid(True)
        plt.text(0.05, 0.95,
                f"Pearson r = {pearson_corr:.4f}\np = {p_val:.2g}",
                transform=plt.gca().transAxes,
                verticalalignment='top',
                fontsize=12,
                bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.6))
        plt.tight_layout()
        plt.savefig("pearson_scatter_stability_transformer.png")
        plt.close()
    
    def save_results(self, y_true, y_pred):
        """保存预测结果到CSV文件"""
        df_result = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred
        })
        df_result.to_csv("pred_vs_actual_stability_transformer.csv", index=False)
        print("📦 保存完毕：best_transformer_model.pth、loss_curve_stability_transformer.png、pearson_scatter_stability_transformer.png、pred_vs_actual_stability_transformer.csv")

    def predict(self, sequences, batch_size=32):
        if self.model is None:
            # 加载checkpoint
            checkpoint = torch.load(self.model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 新格式：包含所有必要参数
                self.scaler = checkpoint['scaler']
                self.encoder = checkpoint['encoder'] 
                self.pad_idx = checkpoint['pad_idx']
                self.max_len = checkpoint['max_len']
                self.model = self.build_model()
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 旧格式：检查是否已有预处理参数
                if not hasattr(self, 'scaler') or not hasattr(self, 'encoder'):
                    raise ValueError("模型文件格式过旧且缺少预处理参数。请先调用preprocess_data()或重新训练模型")
                
                # 从checkpoint中推断模型参数
                state_dict = checkpoint
                # 从位置编码推断max_len
                if 'pos_encoding' in state_dict:
                    checkpoint_max_len = state_dict['pos_encoding'].shape[1]
                    print(f"检测到旧模型的最大长度: {checkpoint_max_len}")
                    # 使用旧模型的参数
                    self.max_len = checkpoint_max_len
                    self.pad_idx = len(self.encoder.classes_)
                
                self.model = self.build_model()
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            
        def extract_codons(seq):
            return [seq[i:i+3] for i in range(0, len(seq), 3) 
                   if len(seq[i:i+3]) == 3 and set(seq[i:i+3]).issubset("ATCG")]
        
        processed_seqs = []
        for seq in sequences:
            seq = seq.upper().replace("U", "T").replace("N", "A")
            codons = extract_codons(seq)
            if len(codons) > 0:
                processed_seqs.append(codons)
            else:
                processed_seqs.append(["ATG"]) 
        
        X_encoded = []
        for seq in processed_seqs:
            try:
                encoded = self.encoder.transform(seq).tolist()
                padded = encoded + [self.pad_idx] * (self.max_len - len(encoded))
                X_encoded.append(padded[:self.max_len])
            except:
                X_encoded.append([self.pad_idx] * self.max_len)
        
        # 批量预测以避免内存溢出
        all_predictions = []
        num_batches = (len(X_encoded) + batch_size - 1) // batch_size
        
        print(f"正在进行批量预测，共 {num_batches} 个批次...")
        
        for i in range(0, len(X_encoded), batch_size):
            batch_end = min(i + batch_size, len(X_encoded))
            batch_X = X_encoded[i:batch_end]
            
            X_tensor = torch.tensor(batch_X, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                batch_preds = self.model(X_tensor).cpu().numpy()
            
            all_predictions.extend(batch_preds)
            
            # 清理GPU内存
            del X_tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"已完成 {i // batch_size + 1}/{num_batches} 个批次")
        
        predictions = self.scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
        return predictions


def run_training(file_path="mRNA_Stability.csv"):
    predictor = RNAStabilityPredictor()
    
    X_tensor, y_tensor, splits = predictor.preprocess_data(file_path)
    
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X_tensor, y_tensor, splits)
    
    train_loader, val_loader, test_loader = predictor.create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    predictor.build_model()
    
    train_losses, val_losses = predictor.train(train_loader, val_loader)
    
    y_true, y_pred, r2, pearson_corr, p_val = predictor.evaluate(test_loader)
    
    predictor.plot_results(train_losses, val_losses, y_true, y_pred, pearson_corr, p_val)
    
    predictor.save_results(y_true, y_pred)
    
    return predictor


if __name__ == "__main__":
    run_training()
