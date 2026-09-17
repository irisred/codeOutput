# charm/grouping.py
from typing import Dict, List, Tuple

def group_by_prefix_at(vis_deltas: List[bytes], depth_bytes: int) -> Dict[bytes, List[int]]:
    """
    用第 n (= depth_bytes) 个字节分组：
      - 组键 = vb[:depth_bytes] + vb[depth_bytes:depth_bytes+1]
        * 若 vb 长度 <= depth_bytes，则 vb[depth_bytes:depth_bytes+1] 为空，键就是 vb[:depth_bytes]
          —— 这类就是 Prefix-Set
    """
    groups: Dict[bytes, List[int]] = {}
    d = int(depth_bytes)
    for i, vb in enumerate(vis_deltas):
        key = vb[:d] + vb[d:d+1]
        groups.setdefault(key, []).append(i)
    return groups

def split_prefix_partial(vis_deltas: List[bytes], idxs: List[int], key: bytes):
    """
    Prefix-Set: vb 恰等于 key
    Partial-Set: vb 以 key 开头且更长
    """
    prefix_idxs, partial_idxs = [], []
    klen = len(key)
    for i in idxs:
        vb = vis_deltas[i]
        if len(vb) == klen and vb == key:
            prefix_idxs.append(i)
        else:
            partial_idxs.append(i)
    return prefix_idxs, partial_idxs