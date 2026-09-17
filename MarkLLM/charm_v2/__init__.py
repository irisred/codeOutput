"""
Next-generation Charm package layout.

We intentionally avoid importing the heavy submodules at package import time,
because several of them depend on `MarkLLM.charm.adapter`, which itself imports
`MarkLLM.charm_v2` pieces. Eager imports here would therefore trigger circular
dependencies once downstream modules (e.g. `llm_wm`) instantiate CharmKGW.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CharmGenerationModel",
    "ByteVocab",
    "CharmByteGenerator",
    "CharmKGW",
    "CharmDetectorV2",
]

_LAZY_IMPORTS = {
    "CharmGenerationModel": "MarkLLM.charm_v2.model",
    "ByteVocab": "MarkLLM.charm_v2.vocab",
    "CharmByteGenerator": "MarkLLM.charm_v2.generator",
    "CharmKGW": "MarkLLM.charm_v2.charm_kgw",
    "CharmDetectorV2": "MarkLLM.charm_v2.detector",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache for future lookups
    return attr
