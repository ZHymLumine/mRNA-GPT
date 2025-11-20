import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.0
    max_len: Optional[int] = None  # if None, auto from data
    embed_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 1024
    num_layers: int = 6
    dropout: float = 0.1
    val_size: float = 0.1


class Vocab:
    def __init__(self, path: str) -> None:
        tokens: List[str] = []
        with open(path, "r") as f:
            for line in f:
                token = line.strip()
                if token:
                    tokens.append(token)
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.cls_id = self.token_to_id[self.cls_token]
        self.sep_id = self.token_to_id[self.sep_token]

    def encode_codons(self, seq: str, add_special: bool = True) -> List[int]:
        seq_u = seq.upper().replace("T", "U")
        codons = [seq_u[i : i + 3] for i in range(0, len(seq_u), 3) if len(seq_u[i : i + 3]) == 3]
        ids: List[int] = [self.token_to_id.get(c, self.unk_id) for c in codons]
        if add_special:
            ids = [self.cls_id] + ids + [self.sep_id]
        return ids

    @property
    def size(self) -> int:
        return len(self.tokens)


class LMDataset(Dataset):
    def __init__(self, input_ids: List[List[int]], pad_id: int, max_len: int) -> None:
        self.pad_id = pad_id
        self.max_len = max_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for ids in input_ids:
            ids = ids[: max_len]
            pad_len = max_len - len(ids)
            x = torch.tensor(ids + [pad_id] * pad_len, dtype=torch.long)
            # Next-token prediction: labels = x shifted left by 1, pad last label
            labels = x.clone()
            labels[:-1] = x[1:]
            labels[-1] = -100  # ignore last position
            # ignore pad positions in loss
            labels = torch.where(x.eq(pad_id), torch.tensor(-100), labels)
            attention_mask = (~x.eq(pad_id)).long()
            self.samples.append((x, attention_mask, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        # bool mask: True 表示禁止注意的位置，与src_key_padding_mask保持同一布尔类型
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [B, L]
        B, L = input_ids.size()
        pad_mask = input_ids.eq(self.pad_id)  # [B, L]
        x = self.tok_emb(input_ids)
        pos = self.pos_emb[:, :L].masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = x + pos
        causal = self._causal_mask(L, input_ids.device)
        hidden = self.transformer(x, mask=causal, src_key_padding_mask=pad_mask)
        hidden = self.dropout(hidden)
        logits = self.lm_head(hidden)
        return logits


def load_checkpoint_weights(model: nn.Module, ckpt_path: str) -> None:
    if not os.path.exists(ckpt_path):
        print(f"[WARN] checkpoint not found: {ckpt_path}; training from scratch.")
        return
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state_dict = state["model_state"]
        else:
            state_dict = state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[CKPT] loaded with missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[WARN] failed to load checkpoint ({e}); training from scratch.")


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # CrossEntropy over vocab, ignore_index=-100 at pad and the last position
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: optim.Optimizer) -> float:
    model.train()
    total = 0.0
    count = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * input_ids.size(0)
        count += input_ids.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels)
        total += loss.item() * input_ids.size(0)
        count += input_ids.size(0)
    return total / max(count, 1)


def plot_curves(train_losses: List[float], val_losses: List[float], out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.title("GPT Fine-tuning on High-Stability mRNA: Train vs Val Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def prepare_datasets(csv_path: str, vocab_path: str, cfg: TrainConfig, stability_threshold: float = 0.5) -> Tuple[Vocab, LMDataset, LMDataset, int]:
    vocab = Vocab(vocab_path)
    df = pd.read_csv(csv_path)
    
    # 支持两种数据格式
    if "Sequence" in df.columns:
        seq_col = "Sequence"
        print(f"使用高稳定性序列数据集，共 {len(df)} 个序列")
        
        # 如果有稳定性信息，可以进一步筛选
        if "Average_Stability" in df.columns:
            original_len = len(df)
            df = df[df["Average_Stability"] >= stability_threshold]
            print(f"筛选平均稳定性 >= {stability_threshold} 的序列：{len(df)}/{original_len}")
        elif "Value" in df.columns and "Predicted" in df.columns:
            original_len = len(df)
            df = df[(df["Value"] >= stability_threshold) & (df["Predicted"] >= stability_threshold)]
            print(f"筛选真实值和预测值都 >= {stability_threshold} 的序列：{len(df)}/{original_len}")
            
    elif "CDS_sequence" in df.columns:
        seq_col = "CDS_sequence"
        print(f"使用传统CDS序列数据集，共 {len(df)} 个序列")
    else:
        raise ValueError("CSV必须包含列: Sequence 或 CDS_sequence")
    
    seqs = df[seq_col].astype(str).tolist()
    
    # 显示稳定性统计信息
    if "Average_Stability" in df.columns:
        print(f"平均稳定性范围: {df['Average_Stability'].min():.3f} - {df['Average_Stability'].max():.3f}")
        print(f"平均稳定性均值: {df['Average_Stability'].mean():.3f}")
    elif "Value" in df.columns:
        print(f"真实稳定性范围: {df['Value'].min():.3f} - {df['Value'].max():.3f}")
        if "Predicted" in df.columns:
            print(f"预测稳定性范围: {df['Predicted'].min():.3f} - {df['Predicted'].max():.3f}")
    
    ids_list = [vocab.encode_codons(s, add_special=True) for s in seqs]
    print(f"序列长度范围: {min(len(ids) for ids in ids_list)} - {max(len(ids) for ids in ids_list)} codons")

    max_len = cfg.max_len or max(len(ids) for ids in ids_list)
    # 为稳妥，限制一个上限（可根据需要调整）
    max_len = min(max_len, 2048)

    train_ids, val_ids = train_test_split(ids_list, test_size=cfg.val_size, random_state=cfg.seed)
    train_dataset = LMDataset(train_ids, vocab.pad_id, max_len)
    val_dataset = LMDataset(val_ids, vocab.pad_id, max_len)
    return vocab, train_dataset, val_dataset, max_len


def save_checkpoint(output_dir: str, model: nn.Module, vocab: Vocab, cfg: TrainConfig, max_len: int, tag: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "vocab": vocab.tokens,
        "config": cfg.__dict__,
        "max_len": max_len,
        "pad_id": vocab.pad_id,
    }
    torch.save(state, os.path.join(output_dir, f"{tag}_model.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GPT on high-stability mRNA sequences via next-token prediction")
    parser.add_argument("--csv", default="/home/acd13855wx/projects/vita/rna2stab/high_stability_sequences.csv")
    parser.add_argument("--vocab", default="/home/acd13855wx/projects/vita/finetune/vocab.txt")
    parser.add_argument("--ckpt", default="/home/acd13855wx/projects/vita/rna2stab/finetune/ckpt_62000.pt")
    parser.add_argument("--outdir", default="/home/acd13855wx/projects/vita/rna2stab/output_finetune_stability_gpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=0, help="0表示自动" )
    parser.add_argument("--stability_threshold", type=float, default=0.5, help="稳定性筛选阈值")

    args = parser.parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        val_size=args.val_size,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        max_len=(None if args.max_len == 0 else args.max_len),
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab, train_dataset, val_dataset, max_len = prepare_datasets(args.csv, args.vocab, cfg, args.stability_threshold)
    model = GPTLanguageModel(
        vocab_size=vocab.size,
        pad_id=vocab.pad_id,
        max_len=max_len,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    # 尝试加载预训练权重
    load_checkpoint_weights(model, args.ckpt)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")

    for epoch in range(cfg.num_epochs):
        tr = train_one_epoch(model, train_loader, device, optimizer)
        vl = evaluate(model, val_loader, device)
        train_losses.append(tr)
        val_losses.append(vl)
        print(f"Epoch {epoch+1}/{cfg.num_epochs} - train_loss: {tr:.6f} - val_loss: {vl:.6f}")

        # 保存最好模型
        if vl < best_val:
            best_val = vl
            save_checkpoint(args.outdir, model, vocab, cfg, max_len, tag="best")

    # 训练完成后保存最后模型、曲线与指标
    save_checkpoint(args.outdir, model, vocab, cfg, max_len, tag="last")
    plot_curves(train_losses, val_losses, os.path.join(args.outdir, "loss_curves.png"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses, "best_val": best_val}, f, indent=2)


if __name__ == "__main__":
    main()


