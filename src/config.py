"""读 config.yaml。参数只有这一处来源。"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]      # financial_ocr/
CONFIG_PATH = ROOT / "config.yaml"


class Cfg(dict):
    """让 cfg.textdet.limit_side 这种写法能用。"""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(f"配置缺少 {key}") from exc
        return Cfg(value) if isinstance(value, dict) else value


def load_config(path: Path | str | None = None) -> Cfg:
    config_path = Path(path) if path else CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return Cfg(data)


def resolve(path: str | Path) -> Path:
    """config.yaml 里的相对路径都相对项目根解析。"""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
