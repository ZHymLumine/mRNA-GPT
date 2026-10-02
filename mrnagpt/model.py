"""Decoder-only codon language model.

A GPT-3 Medium-shaped decoder: 24 layers, d_model 1024, 16 heads, pre-LN, no
bias, GELU, tied embeddings, over a 68-token codon vocabulary and a 2048-codon
context.

The positional encoding is pluggable.  ``rope`` is the default and removes the
hard architectural ceiling on context length; ``learned`` keeps a ``wpe`` table
and is retained so the two can be compared directly in an ablation.

The loss is returned as a **sum** over non-PAD tokens, never a mean.  The caller
owns the denominator, which is what lets gradient accumulation compute an exact
global token mean (see ``train.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .vocab import VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID

# NVIDIA H200 SXM, dense bf16.  The old code hardcoded the A100's 312e12, which
# is why its logs reported an impossible "mfu 91%".
H200_PEAK_TFLOPS = 989e12


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 2048
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    init_std: float = 0.02
    # positional encoding
    pos_encoding: str = "rope"          # "learned" | "rope"
    rope_theta: float = 10000.0
    rope_scaling: str = "none"          # "none" | "linear" | "ntk"
    rope_factor: float = 1.0
    # token ids, carried in the checkpoint so generate.py needs no side config
    pad_token_id: int = PAD_ID
    bos_token_id: int = BOS_ID
    eos_token_id: int = EOS_ID
    peak_tflops: float = H200_PEAK_TFLOPS

    def __post_init__(self):
        if self.pos_encoding not in ("learned", "rope"):
            raise ValueError(f"pos_encoding must be learned|rope, got {self.pos_encoding!r}")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias -- torch's built-in has no such switch."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings with optional inference-time context extension.

    The cos/sin tables are built eagerly for ``max_seq_len`` rather than lazily on
    first use.  A lazy build puts a Python branch on the sequence length inside the
    traced region: the first shape compiled takes the build branch and every later
    one does not, so whichever shape came first recompiles the next time it appears.
    That cost one extra graph per run, always at the bucket where T == block_size.
    Building up front means every training shape traces the identical path.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0,
                 scaling: str = "none", factor: float = 1.0):
        super().__init__()
        if scaling == "ntk":
            # NTK-aware: stretch the base so high-frequency components stay intact.
            theta = theta * factor ** (head_dim / (head_dim - 2))
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # persistent=False keeps checkpoints free of the caches, so rope_scaling /
        # rope_factor can be changed at load time without a state_dict mismatch.
        self.register_buffer("inv_freq", inv, persistent=False)
        self.scaling, self.factor = scaling, factor
        self._build(max_seq_len)

    def _build(self, n: int) -> None:
        pos = torch.arange(n, device=self.inv_freq.device, dtype=torch.float32)
        if self.scaling == "linear":
            pos = pos / self.factor
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def forward(self, T: int, offset: int, device, dtype):
        need = T + offset
        # Only reachable when decoding past block_size (rope_extend); during
        # training need <= block_size always, so the branch never diverges.
        if need > self.cos_cached.size(2):
            self._build(need)
        sl = slice(offset, offset + T)
        return self.cos_cached[:, :, sl].to(dtype), self.sin_cached[:, :, sl].to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class KVCache:
    """Preallocated per-layer key/value cache for autoregressive decoding."""

    def __init__(self, config: GPTConfig, batch_size: int, max_len: int, device, dtype):
        shape = (batch_size, config.n_head, max_len, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.length = 0
        self.max_len = max_len

    def append(self, layer: int, k, v):
        T = k.size(2)
        # length is advanced once per token by GPT.forward, not once per layer
        start = self.length
        self.k[layer][:, :, start:start + T] = k
        self.v[layer][:, :, start:start + T] = v
        return self.k[layer][:, :, :start + T], self.v[layer][:, :, :start + T]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head, self.n_embd = config.n_head, config.n_embd

    def forward(self, x, rope=None, cache: KVCache | None = None, layer_idx: int = 0):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, k, *rope)
        if cache is not None:
            k, v = cache.append(layer_idx, k, v)
            # prefill (cache was empty) is causal; single-token decode is not,
            # because the one query legitimately sees every cached key.
            is_causal = T > 1
        else:
            is_causal = True
        # No padding mask: with right-padding and a causal mask, every real token
        # only attends to real tokens, and PAD hidden states are consumed solely
        # by PAD positions, which never enter the loss.  Passing an explicit
        # additive mask here would also drop us off the fused flash kernel.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=is_causal,
            dropout_p=self.attn_dropout_p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, rope=None, cache: KVCache | None = None):
        x = x + self.attn(self.ln_1(x), rope, cache, self.layer_idx)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.use_rope = config.pos_encoding == "rope"

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
            ln_f=LayerNorm(config.n_embd, config.bias),
        )
        if not self.use_rope:
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rope = (RotaryEmbedding(config.head_dim, config.block_size,
                                     config.rope_theta, config.rope_scaling,
                                     config.rope_factor)
                     if self.use_rope else None)

        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init on residual projections (GPT-2 / nanoGPT)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, 0.0, config.init_std / math.sqrt(2 * config.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, self.config.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, self.config.init_std)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.use_rope:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, cache: KVCache | None = None, pos_offset: int = 0):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        if not self.use_rope:
            pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        rope = self.rope(T, pos_offset, idx.device, x.dtype) if self.use_rope else None
        for block in self.transformer.h:
            x = block(x, rope, cache)
        if cache is not None:
            cache.length += T
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        # SUM, not mean: the caller divides by the global token count.
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                              targets.reshape(-1),
                              ignore_index=self.config.pad_token_id,
                              reduction="sum")
        return logits, nll

    # ------------------------------------------------------------------ #
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Decay 2-D parameters (matmuls, embeddings); never LayerNorm gains."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        opt = torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                eps=1e-8, fused=fused)
        return opt, (len(decay), sum(p.numel() for p in decay),
                     len(no_decay), sum(p.numel() for p in no_decay))

    def flops_per_microbatch(self, B: int, T: int) -> float:
        """Forward+backward FLOPs for an actual (B, T) micro-batch.

        Uses the real bucket width rather than block_size; with length bucketing
        the two differ by ~7x and using block_size would inflate MFU accordingly.
        """
        c = self.config
        n = self.get_num_params(non_embedding=True)
        return 6.0 * n * B * T + 12.0 * c.n_layer * c.n_embd * B * T * T
