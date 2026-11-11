"""财报抓取：巨潮公告 → PDF → 表格页图 → 台账。

**不在主干**：抓数据属于一次性/临时活儿，不是 OCR 推理链路的一部分，按项目约定
（`src/` 只放主干）放在 `tests/crawl/`。入口仍然走 config.yaml 的 `crawl` 段。

模块分层（依赖单向，入口 → 功能 → 内核）：

    __main__  CLI 入口
    runner    编排：目录、并发、预算、止损
    cninfo    公告检索 + 下载（唯一联网的地方）
    pagepick  页筛选 + 渲染（纯 PyMuPDF）
    store     落盘布局 + SQLite 台账
    settings  配置装配
    contracts 数据契约（被所有人 import，不 import 任何人）

对外只暴露入口函数，别 import 内部对象。
"""

from __future__ import annotations

from crawl.contracts import Announcement, CrawlSettings, DocOutcome, PageHit, RenderedPage
from crawl.runner import Harvester, RunReport
from crawl.store import Store

__all__ = [
    "Announcement",
    "CrawlSettings",
    "DocOutcome",
    "PageHit",
    "RenderedPage",
    "Harvester",
    "RunReport",
    "Store",
]
