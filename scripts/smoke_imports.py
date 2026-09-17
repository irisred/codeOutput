#!/usr/bin/env python3
from __future__ import annotations

import importlib


MODULES = [
    "MarkLLM.watermark.bytekgwV6.watermark",
    "MarkLLM.watermark.bytekgwV6.logits_processor",
    "MarkLLM.watermark.bytekgwV6.detector",
    "MarkLLM.watermark.bytekgwV6.prf",
    "MarkLLM.watermark.bytekgwV6.token_bytes",
    "MarkLLM.watermark.kgw.kgw",
    "MarkLLM.watermark.dip.dip",
    "MarkLLM.watermark.unbiased.unbiased",
    "MarkLLM.charm_v2.charm_kgw",
    "MarkLLM.charm_v2.detector",
]


def main() -> None:
    for name in MODULES:
        importlib.import_module(name)
        print(f"OK {name}")


if __name__ == "__main__":
    main()

