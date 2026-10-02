"""LMDB-backed codon dataset with deterministic, length-bucketed, distributed batching.

The published pipeline padded every sequence to ``block_size``.  On the real
archaea length distribution that puts **86% of the compute into PAD** (14% token
efficiency).  Bucketing to 17 geometric widths gets that to 90.5% -- a 6.5x
reduction -- while keeping the number of distinct tensor shapes small enough that
``torch.compile`` never recompiles mid-run.

Everything hinges on one idea: the batching plan is a pure function of
``(seed, epoch, lengths, token_budget, world_size, grad_accum)``.  Every rank
computes a byte-identical plan independently, so there is no sampler
communication, step counts match by construction, ``tokens_seen`` at any step is
known in advance (which makes the LR schedule exactly resumable), and a
mid-epoch resume is a slice rather than a serialised iterator state.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .vocab import PAD_ID, remap_entry

# Length mass sits below 512, so the edges are dense there.  Uniform 128-wide
# buckets give 82% token efficiency; these give 90.5% with one more shape.
BUCKET_EDGES: tuple[int, ...] = (
    64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
    640, 768, 1024, 1280, 1536, 2048,
)


# --------------------------------------------------------------------------- #
# lengths cache
# --------------------------------------------------------------------------- #
def lengths_path_for(lmdb_path: str) -> str:
    return lmdb_path + ".lengths.npy"


def build_lengths(lmdb_path: str, out_path: str | None = None,
                  progress_every: int = 5_000_000) -> np.ndarray:
    """Sequential scan of an LMDB writing ``<lmdb>.lengths.npy`` (uint16).

    ``buffers=True`` makes ``len(v)`` a zero-copy O(1) read, and ``readahead=True``
    (the opposite of the training setting) helps because cursor iteration is
    key-ordered, which for a bulk-loaded LMDB is close to sequential on disk.
    Keys are ``str(i)``, so LMDB's lexicographic order is *not* numeric -- the
    index has to be parsed back out.
    """
    out_path = out_path or lengths_path_for(lmdb_path)
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
                    readahead=True, meminit=False)
    with env.begin(buffers=True) as txn:
        n = txn.stat()["entries"]
        lens = np.zeros(n, dtype=np.uint16)
        seen = 0
        for k, v in txn.cursor():
            # stored length is n_codon + 4; after remap it is n_codon + 2
            lens[int(bytes(k))] = len(v) - 2
            seen += 1
            if progress_every and seen % progress_every == 0:
                print(f"  ... {seen:,}/{n:,}", flush=True)
    env.close()
    if not (lens > 0).all():
        missing = int((lens == 0).sum())
        raise RuntimeError(f"{missing} LMDB keys missing or empty in {lmdb_path}")
    st = os.stat(lmdb_path)
    np.save(out_path, lens)
    with open(out_path + ".meta", "w") as fh:
        fh.write(f"{n}\n{st.st_size}\n")
    return lens


def load_lengths(lmdb_path: str, lengths_path: str | None = None) -> np.ndarray:
    """Load the cache, refusing to silently fall back to an in-job rescan."""
    p = lengths_path or lengths_path_for(lmdb_path)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"missing lengths cache {p}\n"
            f"Build it on a CPU node first:  python -m tools.build_lengths {lmdb_path}\n"
            f"(scanning it here would burn GPU hours re-reading Lustre on every resubmit)")
    lens = np.load(p)
    meta = p + ".meta"
    if os.path.exists(meta):
        with open(meta) as fh:
            want_n, want_size = (int(x) for x in fh.read().split())
        if want_n != lens.shape[0] or want_size != os.stat(lmdb_path).st_size:
            raise RuntimeError(f"stale lengths cache {p}: rebuild it")
    return lens


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
# lmdb refuses a second open of the same file within one process, and two
# datasets legitimately share a path: the train loader and the frozen train
# probe both read train_codon.lmdb.  With num_workers>0 each worker opens its
# own copy and the collision never surfaces, but with num_workers=0 both live
# in the parent.  Environments are therefore cached per (pid, path); the pid is
# part of the key so a forked worker opens its own rather than touching one it
# inherited through the fork.
_ENVS: dict[tuple[int, str], "lmdb.Environment"] = {}


def _open_env(path: str):
    key = (os.getpid(), os.path.realpath(path))
    env = _ENVS.get(key)
    if env is None:
        env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                        readahead=False, meminit=False)
        _ENVS[key] = env
    return env


class CodonLMDBDataset(Dataset):
    """Reads legacy-encoded LMDB entries and remaps them to the 68-token vocab."""

    def __init__(self, lmdb_path: str, lengths_path: str | None = None,
                 check: bool = False):
        self.path = lmdb_path
        self.check = check
        self.env = None                      # opened lazily, after worker fork
        self.lengths = load_lengths(lmdb_path, lengths_path)
        self.n = int(self.lengths.shape[0])

    def _ensure_env(self):
        if self.env is None:
            self.env = _open_env(self.path)

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        idx, T = key
        if idx < 0:                          # sentinel row -> all-PAD
            return None, T
        self._ensure_env()
        with self.env.begin(buffers=True) as txn:
            raw = np.frombuffer(txn.get(b"%d" % idx), dtype=np.uint8)
            # the fancy-index inside remap_entry is what copies the data out of
            # the transaction's buffer, so nothing escapes the txn scope
            ids = remap_entry(raw, check=self.check)
        return ids, T


def collate(batch):
    """Right-pad to the bucket width.  Returns (x, y, n_real_tokens)."""
    if not batch:
        return None
    T = batch[0][1]
    B = len(batch)
    x = np.zeros((B, T), dtype=np.int64)     # PAD_ID == 0
    y = np.zeros((B, T), dtype=np.int64)     # PAD_ID == 0 == ignore_index
    n_real = 0
    for i, (ids, _) in enumerate(batch):
        if ids is None:
            continue
        n = ids.shape[0]
        x[i, :n - 1] = ids[:-1]
        y[i, :n - 1] = ids[1:]
        n_real += n - 1
    return torch.from_numpy(x), torch.from_numpy(y), n_real


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    order: np.ndarray        # int64 (M,)          permuted sequence indices
    mb_start: np.ndarray     # int64 (K,)          offset of micro-batch k into order
    mb_size: np.ndarray      # int32 (K,)
    mb_T: np.ndarray         # int32 (K,)          bucket width
    mb_tokens: np.ndarray    # int64 (K,)          predicted (non-PAD) tokens
    groups: np.ndarray       # int32 (G, ws)       rank-groups; -1 = empty slot
    n_steps: int
    grad_accum: int
    world_size: int
    step_tokens: np.ndarray  # int64 (n_steps,)    global predicted tokens per step
    cum_tokens: np.ndarray   # int64 (n_steps+1,)
    n_seq_used: int
    padded_tokens: int

    @property
    def real_tokens(self) -> int:
        return int(self.step_tokens.sum())

    def efficiency(self) -> float:
        return self.real_tokens / max(self.padded_tokens, 1)


def build_plan(lengths: np.ndarray, token_budget: int, world_size: int,
               grad_accum: int, seed: int, epoch: int,
               edges: tuple[int, ...] = BUCKET_EDGES) -> Plan:
    """Deterministic bucketed batching plan, identical on every rank.

    Two properties are load-bearing:

    * Micro-batches are grouped into rank-groups of ``world_size`` drawn from a
      single bucket, so **all ranks run the same (B, T) in the same micro-step**.
      That removes both the straggler cost of mismatched widths and the
      compile-skew hazard where one rank meets a new shape thousands of steps
      after the others while the rest time out in allreduce.
    * Every sequence appears exactly once per epoch.  Ragged ends are filled with
      sentinel rows (all-PAD, zero loss, zero gradient) rather than dropped, so
      "one epoch" really is one pass -- which is what the single-epoch bacteria
      and eukaryote budgets rest on.  A sentinel row still takes part in the
      forward pass, so no rank ever skips a backward and DDP cannot desync.
    """
    edges_arr = np.asarray(edges, dtype=np.int64)
    L = np.asarray(lengths, dtype=np.int64)
    if L.max() > edges_arr[-1]:
        raise ValueError(f"sequence of length {L.max()} exceeds largest bucket {edges_arr[-1]}")

    rng = np.random.default_rng([seed, epoch])
    perm = rng.permutation(L.shape[0])
    bucket_of = np.searchsorted(edges_arr, L[perm], side="left")

    chunks: list[np.ndarray] = []
    starts: list[int] = []
    sizes: list[int] = []
    widths: list[int] = []
    cursor = 0
    padded_tokens = 0

    for b, T in enumerate(edges_arr):
        sel = perm[bucket_of == b]
        if sel.size == 0:
            continue
        T = int(T)
        B = max(1, token_budget // T)
        n_full = -(-sel.size // B)                       # ceil: keep every sequence
        # pad the bucket's micro-batch count to a multiple of world_size so that
        # rank-groups never straddle two buckets
        n_full += (-n_full) % world_size
        need = n_full * B - sel.size
        if need:
            sel = np.concatenate([sel, np.full(need, -1, dtype=np.int64)])
        chunks.append(sel)
        starts.extend(cursor + j * B for j in range(n_full))
        sizes.extend([B] * n_full)
        widths.extend([T] * n_full)
        cursor += sel.size
        padded_tokens += n_full * B * T

    if not chunks:
        raise RuntimeError("empty plan: token_budget too small or dataset too small")

    order = np.concatenate(chunks)
    mb_start = np.asarray(starts, dtype=np.int64)
    mb_size = np.asarray(sizes, dtype=np.int32)
    mb_T = np.asarray(widths, dtype=np.int32)

    pred = np.where(order >= 0, L[np.maximum(order, 0)] - 1, 0)
    cs = np.concatenate([[0], np.cumsum(pred)])
    mb_tokens = cs[mb_start + mb_size] - cs[mb_start]

    groups = np.arange(mb_start.size, dtype=np.int32).reshape(-1, world_size)
    groups = groups[rng.permutation(groups.shape[0])]

    rem_s = groups.shape[0] % grad_accum
    if rem_s:
        # round the step count up with all-sentinel filler micro-batches rather
        # than discarding whole groups of real data
        T_f = int(edges_arr[0])
        B_f = max(1, token_budget // T_f)
        filler_id = mb_start.size
        order = np.concatenate([order, np.full(B_f, -1, dtype=np.int64)])
        mb_start = np.append(mb_start, cursor)
        mb_size = np.append(mb_size, np.int32(B_f))
        mb_T = np.append(mb_T, np.int32(T_f))
        mb_tokens = np.append(mb_tokens, np.int64(0))
        extra = grad_accum - rem_s
        padded_tokens += extra * world_size * B_f * T_f
        groups = np.concatenate(
            [groups, np.full((extra, world_size), filler_id, dtype=np.int32)])

    n_steps = groups.shape[0] // grad_accum
    if n_steps == 0:
        raise RuntimeError("plan has fewer micro-batch groups than grad_accum")

    flat = groups.reshape(n_steps, grad_accum * world_size)
    step_tokens = mb_tokens[flat].sum(axis=1)

    return Plan(order=order, mb_start=mb_start, mb_size=mb_size, mb_T=mb_T,
                mb_tokens=mb_tokens, groups=groups, n_steps=n_steps,
                grad_accum=grad_accum, world_size=world_size,
                step_tokens=step_tokens.astype(np.int64),
                cum_tokens=np.concatenate([[0], np.cumsum(step_tokens)]).astype(np.int64),
                n_seq_used=int((order >= 0).sum()), padded_tokens=padded_tokens)


class PlanBatchSampler(Sampler):
    """Yields one micro-batch (a list of ``(index, bucket_width)``) at a time."""

    def __init__(self, plan: Plan, rank: int, start_step: int = 0):
        self.p = plan
        self.rank = rank
        self.start_step = start_step

    def __len__(self):
        return (self.p.n_steps - self.start_step) * self.p.grad_accum

    def __iter__(self):
        p = self.p
        for s in range(self.start_step, p.n_steps):
            for j in range(p.grad_accum):
                m = int(p.groups[s * p.grad_accum + j, self.rank])
                o, b, T = int(p.mb_start[m]), int(p.mb_size[m]), int(p.mb_T[m])
                yield [(int(i), T) for i in p.order[o:o + b]]


def make_loader(dataset: CodonLMDBDataset, plan: Plan, rank: int,
                start_step: int = 0, num_workers: int = 8,
                persistent: bool = True):
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_sampler=PlanBatchSampler(plan, rank, start_step),
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        # eval builds a fresh loader every time; persistent workers would leak
        persistent_workers=persistent and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
