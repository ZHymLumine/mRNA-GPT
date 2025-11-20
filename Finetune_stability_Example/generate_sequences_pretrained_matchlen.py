import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch

# 同目录导入GPT定义
from finetune_stability_archaea import Vocab, GPTLanguageModel, set_seed


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    seqs: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA不存在: {path}")
    cur_id = None
    cur_seq_parts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs.append("".join(cur_seq_parts))
                cur_id = line[1:].strip()
                ids.append(cur_id)
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line)
        if cur_id is not None:
            seqs.append("".join(cur_seq_parts))
    return ids, seqs


def compute_codon_lengths(seqs: List[str]) -> np.ndarray:
    return np.array([len(s.strip()) // 3 for s in seqs], dtype=int)


@torch.no_grad()
def sample_with_length_guidance(
    model: GPTLanguageModel,
    vocab: Vocab,
    target_lengths: List[int],
    tol: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> List[List[int]]:
    device = next(model.parameters()).device
    allowed_ids = [i for i, t in enumerate(vocab.tokens) if len(t) == 3 and set(t).issubset({"A", "U", "C", "G"})]
    start_id = vocab.token_to_id.get("AUG")
    stop_ids = [vocab.token_to_id[s] for s in ("UAA", "UAG", "UGA") if s in vocab.token_to_id]
    if start_id is None or not stop_ids:
        raise ValueError("vocab需包含AUG与(UAA/UAG/UGA)之一")

    sequences: List[List[int]] = []
    for tgt_len in target_lengths:
        lower = max(1, tgt_len - tol)
        upper = tgt_len + tol
        tokens = [vocab.cls_id, start_id]
        # 已生成codon数（不含CLS）：len(tokens) - 1
        while (len(tokens) - 1) < upper and len(tokens) < (model.max_len - 2):
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)[:, -1, :] / max(temperature, 1e-6)

            # 屏蔽特殊与非三联体
            disallow = torch.ones_like(logits, dtype=torch.bool)
            disallow[:, allowed_ids] = False
            logits = logits.masked_fill(disallow, float('-inf'))

            cur_codons = len(tokens) - 1
            # 达到下界前禁止终止；处于范围内允许终止；超出上界强制终止
            if cur_codons < lower:
                logits[:, stop_ids] = float('-inf')
            elif cur_codons > upper:
                # 仅允许终止，尽快结束
                keep = torch.full_like(logits, float('-inf'))
                keep[:, stop_ids] = 0.0
                logits = keep
            # 永久禁止PAD/SEP
            logits[:, vocab.sep_id] = float('-inf')
            logits[:, vocab.pad_id] = float('-inf')

            # 可选top-k
            if top_k > 0:
                values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                keep = torch.full_like(logits, float('-inf'))
                keep.scatter_(1, indices, values)
                logits = keep

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            # 安全：如采到非法id，跳过重采（极少发生）
            if next_id not in allowed_ids:
                continue
            # 合法终止
            if next_id in stop_ids and cur_codons >= lower:
                tokens.append(next_id)
                break
            tokens.append(next_id)

        # 若未终止，则强制终止
        if tokens[-1] not in stop_ids:
            tokens.append(random.choice(stop_ids))
        tokens.append(vocab.sep_id)
        sequences.append(tokens)
    return sequences


def decode_to_dna(ids: List[int], vocab: Vocab) -> str:
    toks = []
    for tid in ids:
        if tid in (vocab.pad_id, vocab.cls_id, vocab.sep_id):
            continue
        toks.append(vocab.tokens[tid])
    return ''.join(toks).replace('U', 'T')


def main() -> None:
    parser = argparse.ArgumentParser(description="Use pretrained GPT to generate sequences matching finetuned length distribution")
    parser.add_argument("--ckpt", default="/home/acd13855wx/projects/vita/rna2stab/finetune/ckpt_62000.pt")
    parser.add_argument("--vocab", default="/home/acd13855wx/projects/vita/rna2stab/finetune/vocab.txt")
    parser.add_argument("--ref_fasta", default="", help="参考长度分布的FASTA（可选，留空则使用随机长度分布）")
    parser.add_argument("--outdir", default="/home/acd13855wx/projects/vita/rna2stab/finetune/pretrained_generated_matchlen")
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--tol", type=int, default=10, help="与目标长度的容忍（单位：codon）")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # 结构参数（用于仅权重ckpt），若ckpt提供model_args会覆盖
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model_max_len", type=int, default=2048)
    args = parser.parse_args()

    set_seed(args.seed)

    # 读取参考长度分布或使用默认分布
    if args.ref_fasta and os.path.exists(args.ref_fasta):
        print(f"使用参考FASTA文件: {args.ref_fasta}")
        _, ref_seqs = read_fasta(args.ref_fasta)
        ref_lengths = compute_codon_lengths(ref_seqs)
        if ref_lengths.size == 0:
            raise ValueError("参考FASTA中无序列")
        targets = np.random.choice(ref_lengths, size=args.num, replace=True).tolist()
    else:
        print("使用默认长度分布（100-1000 codons）")
        # 使用合理的长度分布：大多数在200-600 codons之间
        targets = np.random.choice(
            np.arange(100, 1001), 
            size=args.num, 
            replace=True,
            p=np.concatenate([
                np.linspace(0.1, 1.0, 100),  # 100-199: 递增
                np.ones(400),                # 200-599: 平稳
                np.linspace(1.0, 0.1, 401)   # 600-1000: 递减
            ]) / np.sum(np.concatenate([
                np.linspace(0.1, 1.0, 100),
                np.ones(400),
                np.linspace(1.0, 0.1, 401)
            ]))
        ).tolist()

    # 加载vocab
    tokens: List[str] = []
    with open(args.vocab, 'r') as vf:
        for line in vf:
            tok = line.strip()
            if tok:
                tokens.append(tok)
    vocab = Vocab.__new__(Vocab)
    vocab.tokens = tokens
    vocab.token_to_id = {t: i for i, t in enumerate(tokens)}
    vocab.pad_token = "[PAD]"; vocab.pad_id = vocab.token_to_id[vocab.pad_token]
    vocab.unk_token = "[UNK]"; vocab.unk_id = vocab.token_to_id[vocab.unk_token]
    vocab.cls_token = "[CLS]"; vocab.cls_id = vocab.token_to_id[vocab.cls_token]
    vocab.sep_token = "[SEP]"; vocab.sep_id = vocab.token_to_id[vocab.sep_token]
    vocab.mask_token = "[MASK]"

    # 构建模型并尽量从ckpt提取结构参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location="cpu")
    n_embd = args.embed_dim; n_head = args.heads; n_layer = args.layers; block_size = args.model_max_len
    if isinstance(state, dict) and 'model_args' in state and isinstance(state['model_args'], dict):
        ma = state['model_args']
        n_embd = int(ma.get('n_embd', n_embd))
        n_head = int(ma.get('n_head', n_head))
        n_layer = int(ma.get('n_layer', n_layer))
        block_size = int(ma.get('block_size', block_size))
    ff_dim = args.ff_dim if args.ff_dim else 4 * n_embd

    model = GPTLanguageModel(
        vocab_size=len(tokens), pad_id=vocab.pad_id, max_len=block_size,
        embed_dim=n_embd, num_heads=n_head, ff_dim=ff_dim,
        num_layers=n_layer, dropout=args.dropout
    ).to(device)

    state_for_model = state.get('model_state') if isinstance(state, dict) else state
    if state_for_model is None and isinstance(state, dict) and 'model' in state:
        state_for_model = state['model']
    missing, unexpected = model.load_state_dict(state_for_model, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict not exact: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # 引导长度生成
    os.makedirs(args.outdir, exist_ok=True)
    seq_ids = sample_with_length_guidance(
        model, vocab, target_lengths=targets, tol=args.tol,
        temperature=args.temperature, top_k=args.top_k
    )

    # 写FASTA（DNA字母）
    out_fa = os.path.join(args.outdir, 'generated_sequences_matchlen.fasta')
    with open(out_fa, 'w') as f:
        for i, ids in enumerate(seq_ids):
            dna = decode_to_dna(ids, vocab)
            f.write(f">seq_{i+1}\n"); f.write(dna + "\n")
    print(f"Saved {len(seq_ids)} sequences to {out_fa}")


if __name__ == '__main__':
    main()



