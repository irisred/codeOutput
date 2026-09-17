# charm/types.py
from dataclasses import dataclass
from typing import List
import torch

@dataclass
class Candidate:
    token_id: int
    logit: float
    vis_delta: bytes

@dataclass
class State:
    input_ids: torch.Tensor
    vis_ctx: bytes
    round_id: int = 0
