from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class PRFTraceRecord:
    phase: str
    token_index: int
    token_tail: List[int]
    byte_pos: int
    prefix_bytes_hex: str
    greenlist: List[int]
    extra: Dict[str, object] = field(default_factory=dict)


class PRFTracer:
    def __init__(self, *, limit: int | None = None, token_tail: int = 32) -> None:
        """
        limit: None 或 <=0 表示不限制记录条数；>0 时按条数裁剪。
        token_tail: 记录上下文时保留的 token 尾部长度。
        """
        if limit is None or int(limit) <= 0:
            self.limit = None  # 不限制
        else:
            self.limit = int(limit)
        self.token_tail = max(1, int(token_tail))
        self.records: Dict[str, List[PRFTraceRecord]] = {"gen": [], "det": []}

    def reset(self) -> None:
        for key in self.records:
            self.records[key].clear()

    def phase_records(self, phase: str) -> List[PRFTraceRecord]:
        return self.records.setdefault(phase, [])

    def log(
        self,
        phase: str,
        *,
        token_index: int,
        token_ids: Sequence[int],
        prefix_bytes: bytes,
        byte_pos: int,
        greenlist: Sequence[int],
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        bucket = self.phase_records(phase)
        if self.limit is not None and len(bucket) >= self.limit:
            return
        token_ids_list = [int(x) for x in token_ids]
        tail = token_ids_list[-self.token_tail :] if token_ids_list else []
        rec = PRFTraceRecord(
            phase=phase,
            token_index=int(token_index),
            token_tail=tail,
            byte_pos=int(byte_pos),
            prefix_bytes_hex=bytes(prefix_bytes).hex(),
            greenlist=[int(x) for x in greenlist],
            extra=dict(extra or {}),
        )
        bucket.append(rec)


__all__ = ["PRFTracer", "PRFTraceRecord"]
