"""アダプタ registry: models.yaml のキーからアダプタを生成・キャッシュする。"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

from .base import BaseAdapter

_REGISTRY = {}


def register(kind: str):
    def deco(cls):
        _REGISTRY[kind] = cls
        return cls
    return deco


def _ensure_registered():
    if _REGISTRY:
        return
    from .jev_ja import JevJaAdapter
    from .open_jev import OpenJevAdapter
    from .gliner2 import Gliner2Adapter
    from .nli_zeroshot import NliZeroShotAdapter
    from .llm_verbalized import LlmVerbalizedAdapter
    from .crossenc_hf import CrossEncHfAdapter
    from .tiny_jev import TinyJevAdapter
    from .jev_style import JevStyleAdapter
    from .laya import LayaAdapter
    _REGISTRY.update({"crossenc_hf": CrossEncHfAdapter, "tiny_jev": TinyJevAdapter, "jev_style": JevStyleAdapter, "laya": LayaAdapter})
    _REGISTRY.update({
        "jev_ja": JevJaAdapter, "open_jev": OpenJevAdapter, "gliner2": Gliner2Adapter,
        "nli_zeroshot": NliZeroShotAdapter, "llm_verbalized": LlmVerbalizedAdapter,
    })


CONFIG_PATH = os.environ.get("JEV_LAB_CONFIG", os.path.join(os.path.dirname(__file__), "..", "..", "config", "models.yaml"))


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_cache: dict[str, BaseAdapter] = {}


def get_adapter(name: str | None = None) -> BaseAdapter:
    _ensure_registered()
    cfg = load_config()
    name = name or cfg["default"]
    if name in _cache:
        return _cache[name]
    mcfg = cfg["models"][name]
    ad = _REGISTRY[mcfg["adapter"]](name, mcfg)
    _cache[name] = ad
    return ad


def list_models() -> list[dict]:
    cfg = load_config()
    return [{"name": k, **v} for k, v in cfg["models"].items()]
