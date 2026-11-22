"""配置加载：YAML + 点号路径访问 + 相对路径解析到项目根目录。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, data: dict, root: Path = ROOT):
        self._d = data or {}
        self.root = root

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self._d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        q = Path(p)
        return q if q.is_absolute() else self.root / q


def load(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data or {}, root=cfg_path.resolve().parent)
