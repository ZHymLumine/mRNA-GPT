import argparse
import os
import random
from typing import List

import numpy as np
import torch

# 与同目录下的finetune_stability_archaea.py相对导入
from finetune_stability_archaea import Vocab, GPTLanguageModel, set_seed


@torch.no_grad()
def sample_sequences(
    model: GPTLanguageModel,
    vocab: Vocab,
    num_sequences: int,
    min_len: int,
    max_len: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> List[List[int]]:
    device = next(model.parameters()).device
    # 仅允许三联体碱基token作为采样空间
    allowed_ids = [i for i, t in enumerate(vocab.tokens) if len(t) == 3 and set(t).issubset({"A", "U", "C", "G"})]
    # 起始/终止密码子索引（RNA字母表）
    start_tok = "AUG"
    stop_toks = {"UAA", "UAG", "UGA"}
    if start_tok not in vocab.token_to_id:
        raise ValueError("vocab中缺少AUG起始密码子")
    start_id = vocab.token_to_id[start_tok]
    stop_ids = [vocab.token_to_id[s] for s in stop_toks if s in vocab.token_to_id]
    if len(stop_ids) == 0:
        raise ValueError("vocab中缺少任何终止密码子(UAA/UAG/UGA)")
    sequences: List[List[int]] = []
    for _ in range(num_sequences):
        target_len = random.randint(min_len, max_len)
        # 强制以AUG起始
        tokens = [vocab.cls_id, start_id]
        while (len(tokens) - 1) < target_len and len(tokens) < (model.max_len - 2):
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)[:, -1, :]  # [1, V]
            logits = logits / max(temperature, 1e-6)
            # 屏蔽所有非三联体碱基token与特殊标记
            mask_disallow = torch.ones_like(logits, dtype=torch.bool)
            mask_disallow[:, allowed_ids] = False
            logits = logits.masked_fill(mask_disallow, float('-inf'))
            # 在达到最小长度之前，禁止采样终止codon
            if (len(tokens) - 1) < min_len:
                logits[:, stop_ids] = float('-inf')
            # 永久禁止PAD/SEP
            logits[:, vocab.sep_id] = float('-inf')
            logits[:, vocab.pad_id] = float('-inf')
            # 可选top-k截断
            if top_k > 0:
                values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, indices, values)
                logits = mask
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            # 安全检查：若采样到非法id则跳过本次（极少数数值异常）
            if next_id not in allowed_ids:
                continue
            # 若采样到终止且长度已达标，则接受并跳出内部循环
            if next_id in stop_ids and (len(tokens) - 1) >= min_len:
                tokens.append(next_id)
                break
            tokens.append(next_id)
        # 若未以终止codon结尾，强制追加一个终止codon
        if tokens[-1] not in stop_ids:
            tokens.append(random.choice(stop_ids))
        # 末尾追加SEP
        tokens.append(vocab.sep_id)
        sequences.append(tokens)
    return sequences


def decode_codons(ids: List[int], vocab: Vocab) -> str:
    # 去掉头尾特殊符号
    toks = []
    for tid in ids:
        if tid in (vocab.pad_id, vocab.cls_id, vocab.sep_id):
            continue
        toks.append(vocab.tokens[tid])
    # 将U替换回T，拼接形成DNA形式的CDS
    codon_seq = ''.join(toks)
    return codon_seq.replace('U', 'T')


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sequences using GPT model (fine-tuned or pretrained)")
    parser.add_argument("--ckpt", default="/home/acd13855wx/projects/vita/rna2stab/output_finetune_stability_gpt/best_model.pt")
    parser.add_argument("--outdir", default="/home/acd13855wx/projects/vita/rna2stab/finetune/finetune_generated")
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--min_len", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # 兼容仅权重checkpoint：提供可选vocab与模型结构参数
    parser.add_argument("--vocab", default="/home/acd13855wx/projects/vita/rna2stab/finetune/vocab.txt", help="可选，若ckpt无vocab则从文件加载")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model_max_len", type=int, default=2048, help="模型最大长度（若ckpt未提供）")
    args = parser.parse_args()

    set_seed(args.seed)

    # 加载checkpoint
    state = torch.load(args.ckpt, map_location="cpu")
    # 1) 处理vocab
    if isinstance(state, dict) and "vocab" in state:
        tokens: List[str] = state["vocab"]
    else:
        # 从vocab.txt加载
        tokens = []
        with open(args.vocab, "r") as vf:
            for line in vf:
                tok = line.strip()
                if tok:
                    tokens.append(tok)
    # 2) 处理max_len
    if isinstance(state, dict) and "max_len" in state:
        max_len: int = int(state["max_len"])  # 模型的最大长度
    else:
        max_len = int(args.model_max_len)
    # 3) 处理结构超参
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    embed_dim = int(cfg.get("embed_dim", args.embed_dim))
    num_heads = int(cfg.get("num_heads", args.heads))
    ff_dim = int(cfg.get("ff_dim", args.ff_dim))
    num_layers = int(cfg.get("num_layers", args.layers))
    dropout = float(cfg.get("dropout", args.dropout))

    # 构建Vocab（从tokens列表构造实例）
    vocab = Vocab.__new__(Vocab)
    vocab.tokens = tokens
    vocab.token_to_id = {t: i for i, t in enumerate(tokens)}
    vocab.pad_token = "[PAD]"
    vocab.unk_token = "[UNK]"
    vocab.cls_token = "[CLS]"
    vocab.sep_token = "[SEP]"
    vocab.mask_token = "[MASK]"
    vocab.pad_id = vocab.token_to_id[vocab.pad_token]
    vocab.unk_id = vocab.token_to_id[vocab.unk_token]
    vocab.cls_id = vocab.token_to_id[vocab.cls_token]
    vocab.sep_id = vocab.token_to_id[vocab.sep_token]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTLanguageModel(
        vocab_size=len(tokens),
        pad_id=vocab.pad_id,
        max_len=max_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    # 兼容仅权重或包含model_state两种格式
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])  # type: ignore[index]
    else:
        model.load_state_dict(state)
    model.eval()

    # 采样序列
    os.makedirs(args.outdir, exist_ok=True)
    seq_ids = sample_sequences(
        model, vocab, num_sequences=args.num, min_len=args.min_len, max_len=args.max_len,
        temperature=args.temperature, top_k=args.top_k
    )

    # 写出为FASTA
    out_fa = os.path.join(args.outdir, "generated_sequences.fasta")
    with open(out_fa, "w") as f:
        for i, ids in enumerate(seq_ids):
            dna_seq = decode_codons(ids, vocab)
            f.write(f">seq_{i+1}\n")
            f.write(dna_seq + "\n")
    print(f"Saved {len(seq_ids)} sequences to {out_fa}")


if __name__ == "__main__":
    main()


