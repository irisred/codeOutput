# charm/config.py
class CharmConfig:
    def __init__(
        self,
        enabled: bool = True,
        normalize_form: str = "NONE",
        prefix_length: int = 0,
        use_unbiased_prf: bool = True,
        use_pos_salt: bool = False,
        debug_token_dump: int = 0,
        debug_full_byte_probs: bool = False,
        debug_topk: int = 8,
        debug_list_ops: bool = False,
        visible_bias_only: bool = False,
    ):
        self.enabled = enabled
        self.normalize_form = normalize_form
        # 固定采用 256 字节域，去掉可配置项，默认启用
        self.fixed_256_bytes = True
        # 使用“哈希到集合”的无偏 PRF（HMAC(window_digest || b) < gamma）
        self.use_unbiased_prf = bool(use_unbiased_prf)
        # 是否使用位置盐（默认关闭）
        self.use_pos_salt = bool(use_pos_salt)
        self.prefix_length = int(prefix_length)
        self.debug_token_dump = int(debug_token_dump)
        self.debug_full_byte_probs = bool(debug_full_byte_probs)
        self.debug_topk = int(debug_topk)
        self.debug_list_ops = bool(debug_list_ops)
        self.visible_bias_only = bool(visible_bias_only)
