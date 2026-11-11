"""config.yaml 的 crawl 段 → CrawlSettings。只做装配和校验，不做 IO。"""

from __future__ import annotations

import logging

from crawl.contracts import CrawlSettings

log = logging.getLogger(__name__)

DEFAULTS = {
    "data_root": "data",
    "workers": 4,
    "page_dpi": 200,
    "min_rows": 5,
    "max_pages_per_doc": 8,
    "max_pdf_mb": 60,
    "pdf_budget_gb": 60.0,
    "min_free_gb": 20.0,
    "target_images": 30000,
    "max_docs": 0,
    "timeout_s": 60.0,
    "retries": 3,
    "retry_wait_s": 2.0,
    "min_delay_s": 0.15,
}


def build_settings(cfg: dict, resolve) -> CrawlSettings:
    """cfg 是整份 config.yaml；缺 crawl 段就用默认值（R18：入口处校验一次）。"""
    raw = dict(DEFAULTS)
    raw.update((cfg or {}).get("crawl") or {})
    data_root = resolve(raw.pop("data_root"))
    if not str(data_root).strip():
        raise ValueError("crawl.data_root 不能为空")
    settings = CrawlSettings(data_root=data_root, **raw)
    if settings.workers < 1:
        raise ValueError(f"crawl.workers 必须 >=1，收到 {settings.workers}")
    if settings.min_rows < 2:
        raise ValueError(f"crawl.min_rows 太小（{settings.min_rows}），会把正文页当表格")
    log.debug("crawl 配置：%s", settings)
    return settings
