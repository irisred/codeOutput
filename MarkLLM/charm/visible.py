# watermark/charm/visible.py
from __future__ import annotations
from typing import Dict, Optional
from MarkLLM.charm_v2.token_bytes import TokenByteVocab

_TOKEN_VOCABS: Dict[int, TokenByteVocab] = {}


def _get_token_byte_vocab(tokenizer) -> TokenByteVocab:
    """
    获取并缓存 TokenByteVocab，确保跨 tokenizer 类型的鲁棒性。
    支持 ByteLevelBPE / SentencePiece / WordPiece。
    """
    key = id(tokenizer)
    vocab = _TOKEN_VOCABS.get(key)
    if vocab is not None:
        return vocab

    # --- 安全识别 tokenizer 类型 ---
    name = getattr(tokenizer, "__class__", type(tokenizer)).__name__.lower()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    is_bytelevel = (
        "gpt2" in name or "bpe" in name or
        "bytelevel" in name or
        (backend is not None and "ByteLevel" in str(type(backend)))
    )

    # --- 构造 EOS / 特殊 token 列表 ---
    end_token_ids = []
    for name_attr in ["eos_token_id", "sep_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, name_attr, None)
        if tid is not None:
            try:
                end_token_ids.append(int(tid))
            except Exception:
                pass

    # --- 构造映射 ---
    try:
        vocab = TokenByteVocab.from_tokenizer(
            tokenizer,
            end_token_ids=end_token_ids,
        )
    except Exception as e:
        # fallback 安全模式：构建空表并打印警告
        print(f"[warn] TokenByteVocab init failed: {e}. Falling back to minimal mapping.")
        vocab = TokenByteVocab(mapping={}, vocab_size=getattr(tokenizer, "vocab_size", 65536))

    _TOKEN_VOCABS[key] = vocab
    return vocab


def to_visible_bytes(tokenizer, token_id: int) -> bytes:
    """
    将单个 token id 转为可见 bytes，兼容各种 tokenizer。
    保证无 silent drop。
    """
    vocab = _get_token_byte_vocab(tokenizer)
    try:
        data = vocab.token_bytes(int(token_id))
        if not data:
            # fallback: 尝试直接 decode
            token_str: Optional[str] = None
            try:
                pieces = tokenizer.convert_ids_to_tokens([int(token_id)], skip_special_tokens=False)
                if pieces:
                    token_str = pieces[0]
            except Exception:
                pass

            if token_str is None:
                try:
                    token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False)
                except Exception:
                    token_str = f"<UNK_{token_id}>"

            # ⚠️ 不再使用 errors="ignore"，避免丢失字节
            data = str(token_str).encode("utf-8", errors="backslashreplace")
        return data
    except Exception as e:
        print(f"[warn] to_visible_bytes({token_id}) failed: {e}")
        return f"<ERR_{token_id}>".encode("utf-8")


def text_to_visible_bytes(text: str) -> bytes:
    """
    文本 → 可见 bytes，保留所有非标准字符。
    """
    if not isinstance(text, str):
        text = str(text)
    # ⚠️ 不使用 ignore，确保非法字符可追踪
    return text.encode("utf-8", errors="backslashreplace")
