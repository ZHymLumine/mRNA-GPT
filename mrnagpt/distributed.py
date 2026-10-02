"""Minimal DDP helpers."""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


class DDPInfo:
    def __init__(self):
        self.enabled = int(os.environ.get("RANK", -1)) != -1
        if self.enabled:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.rank = self.local_rank = 0
            self.world_size = 1
        self.is_master = self.rank == 0

    def init(self, backend: str = "nccl"):
        if self.enabled:
            # A generous timeout for the first epoch: torch.compile warms up 17
            # bucket shapes and a rank that meets one late would otherwise trip
            # the NCCL watchdog while the others sit in allreduce.
            dist.init_process_group(backend=backend,
                                    timeout=datetime.timedelta(minutes=60))
            torch.cuda.set_device(self.local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(0)
        return self

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"

    def all_reduce(self, t: torch.Tensor, op=None):
        if self.enabled:
            dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
        return t

    def broadcast_flag(self, value: bool) -> bool:
        if not self.enabled:
            return value
        t = torch.tensor([1 if value else 0], device=self.device, dtype=torch.int32)
        dist.broadcast(t, src=0)
        return bool(t.item())

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def shutdown(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
