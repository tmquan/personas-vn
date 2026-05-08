"""Tiny YAML config loader.

Loads any YAML file into a ``Config`` object that supports both attribute
access (``cfg.fetch.per_page``) and dict-style access (``cfg["fetch"]``). We
deliberately avoid pulling in ``omegaconf`` / ``hydra`` to keep the dep
footprint small; the configs we manage here are flat and simple.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from packages.common.paths import resolve


class Config(Mapping[str, Any]):
    """Read-only dotted-access wrapper around a nested dict."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        # Recursively wrap nested dicts so attribute access works at any depth.
        self._data: dict[str, Any] = {
            k: Config(v) if isinstance(v, Mapping) else v
            for k, v in (data or {}).items()
        }

    # ----- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    # ----- Attribute access -------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ----- Conversion -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._data.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a :class:`Config`."""
    full = resolve(path)
    if not full.exists():
        raise FileNotFoundError(f"Config not found: {full}")
    with full.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {full}")
    return Config(data)
