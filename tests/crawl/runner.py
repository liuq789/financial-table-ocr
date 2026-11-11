"""抓取编排：枚举目录 → 并发下载 → 挑页渲图 → 记账。

只做编排，不碰网络细节（cninfo）、不碰磁盘细节（store）、不碰解析细节（pagepick）。
三层各自能单独跑，这里负责把它们串成一条带预算和护栏的流水线。

三条止损线，任一触发就停（不会跑成"把盘撑爆"）：
    * 页图数达到 target_images
    * 磁盘剩余低于 min_free_gb
    * 没有待办任务了

单独跑：
    PYTHONPATH=src:tests python -m crawl.runner --kind annual --years 2019-2024 --limit 50
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from crawl.cninfo import CninfoClient, FetchError
from crawl.contracts import Announcement, CrawlSettings, DocOutcome
from crawl.pagepick import extract_pages
from crawl.store import Store

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    docs: int = 0
    images: int = 0
    failed: int = 0
    skipped: int = 0
    bytes_pdf: int = 0
    stopped_by: str = ""
    errors: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"完成 {self.docs} 份 / 出图 {self.images} 张 / 失败 {self.failed} / "
            f"跳过 {self.skipped} / PDF {self.bytes_pdf / 2**30:.1f}GB / 停止原因 {self.stopped_by}"
        )


class Harvester:
    def __init__(self, cfg: CrawlSettings, store: Store, client: CninfoClient) -> None:
        self.cfg = cfg
        self.store = store
        self.client = client
        self._pdf_bytes = store.stats()["bytes_pdf"]
        self._images_done = store.stats()["images"]   # 已入库的页图数（target 按总量算）
        self._consec_fail = 0        # 连续失败数，用来判断"是不是被 ban 了"

    # ---------------------------------------------------------------- 目录
    def catalog(self, kind: str, years: list[str], limit: int = 0) -> int:
        """把公告元数据写进台账。不下载，只登记。返回新增条数。"""
        added = 0
        for year in years:
            for ann in self.client.iter_announcements(kind, year, limit=limit):
                added += int(self.store.add(ann))
            log.info("目录 %s %s：累计新增 %d 条", kind, year, added)
        return added

    # ---------------------------------------------------------------- 主循环
    def run(self, max_docs: int = 0, target_images: int | None = None) -> RunReport:
        cfg = self.cfg
        target = cfg.target_images if target_images is None else target_images
        limit = max_docs or cfg.max_docs
        report = RunReport()
        started = time.time()

        while True:
            stop = self._stop_reason(report, limit, target)
            if stop:
                report.stopped_by = stop
                break
            batch = self.store.pending(limit=self._batch_size(limit - report.docs))
            if not batch:
                report.stopped_by = "没有待办任务"
                break
            self._run_batch(batch, report, limit, target, started)

        report.bytes_pdf = self._pdf_bytes
        log.info("抓取结束：%s", report.line())
        return report

    def _batch_size(self, remaining: int) -> int:
        """一批最多 workers*2；有份数上限时不许越过上限。"""
        size = self.cfg.workers * 2
        return size if remaining <= 0 else min(size, max(remaining, 1))

    def _stop_reason(self, report: RunReport, limit: int, target: int) -> str:
        if limit and report.docs >= limit:
            return f"达到份数上限 {limit}"
        if self._images_done >= target:
            return f"达到页图目标 {target}（当前 {self._images_done}）"
        if self._consec_fail >= 20:
            return f"连续失败 {self._consec_fail} 份，疑似被限流/断网，先停下"
        free = self.store.free_gb()
        if free < self.cfg.min_free_gb:
            return f"磁盘剩余 {free:.1f}GB 低于保底 {self.cfg.min_free_gb}GB"
        return ""

    def _run_batch(
        self, batch: list[Announcement], report: RunReport, limit: int, target: int, started: float
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures = {pool.submit(self._process, ann): ann for ann in batch}
            for fut in as_completed(futures):
                ann = futures[fut]
                try:
                    outcome = fut.result()
                except _Skip as exc:
                    report.skipped += 1
                    self._consec_fail = 0
                    self.store.mark_failed(ann.key, f"skip: {exc}")
                    log.info("跳过 %s：%s", ann.slug, exc)
                    continue
                except Exception as exc:                        # noqa: BLE001 单份失败不影响全局
                    report.failed += 1
                    self._consec_fail += 1
                    report.errors.append(f"{ann.slug}: {exc}")
                    self.store.mark_failed(ann.key, f"{type(exc).__name__}: {exc}")
                    log.warning("失败 %s：%s", ann.slug, exc)
                    continue
                report.docs += 1
                self._consec_fail = 0
                report.images += outcome.n_images
                self._images_done += outcome.n_images
                if outcome.kept_pdf:
                    self._pdf_bytes += outcome.pdf_bytes
                self.store.mark_done(outcome)
                if report.docs % 10 == 0:
                    self._log_progress(report, started, target)

    def _process(self, ann: Announcement) -> DocOutcome:
        cfg = self.cfg
        if ann.size_kb > cfg.max_pdf_mb * 1024:
            raise _Skip(f"{ann.size_kb / 1024:.0f}MB 超过单份上限 {cfg.max_pdf_mb}MB")
        self.store.mark_running(ann.key)

        t0 = time.time()
        data = self.client.download(ann)
        if len(data) > cfg.max_pdf_mb * 2**20:
            raise _Skip(f"实际 {len(data) / 2**20:.0f}MB 超过上限")

        n_pages, pages = extract_pages(
            data, min_rows=cfg.min_rows, max_pages=cfg.max_pages_per_doc, dpi=cfg.page_dpi
        )
        if not pages:
            raise _Skip(f"{n_pages} 页里没有表格页")

        kept = self._pdf_bytes < cfg.pdf_budget_gb * 2**30
        if kept:
            self.store.save_pdf(ann, data)
        else:
            self.store.drop_pdf(ann)
        self.store.save_pages(ann, n_pages, pages)
        outcome = DocOutcome(
            ann=ann, n_pages=n_pages, pages=pages, elapsed_s=time.time() - t0,
            kept_pdf=kept, pdf_bytes=len(data) if kept else 0,
        )
        log.debug("%s %d 页中留 %d 页，%.1fs", ann.slug, n_pages, outcome.n_images, outcome.elapsed_s)
        return outcome

    def _log_progress(self, report: RunReport, started: float, target: int) -> None:
        spent = max(time.time() - started, 1e-6)
        rate = report.docs / spent * 60
        eta_docs = max(target - self._images_done, 0) / max(report.images / report.docs, 0.1)
        log.info(
            "进度 +%d 份 / +%d 张（累计 %d 张），%.1f 份每分钟，磁盘剩 %.1fGB，还需约 %.0f 份",
            report.docs, report.images, self._images_done, rate, self.store.free_gb(), eta_docs,
        )


class _Skip(Exception):
    """这份不用处理（超大 / 没表格页），算跳过不算失败。"""


def _main() -> int:
    import argparse

    from config import load_config, resolve
    from crawl.settings import build_settings
    from logging_setup import setup_logging

    ap = argparse.ArgumentParser(description="批量抓取（先 catalog 再 run）")
    ap.add_argument("--kind", default="annual", choices=["annual", "semiannual"])
    ap.add_argument("--years", default="", help="披露年份，逗号分隔或 2019-2024；空=不枚举")
    ap.add_argument("--limit", type=int, default=0, help="每个年份最多登记几条")
    ap.add_argument("--docs", type=int, default=0, help="本次最多处理几份")
    ap.add_argument("--target", type=int, default=None, help="页图目标数")
    args = ap.parse_args()

    setup_logging()
    cfg = build_settings(load_config(), resolve)
    with Store(cfg) as store:
        harvester = Harvester(cfg, store, CninfoClient(cfg))
        if args.years:
            harvester.catalog(args.kind, _years(args.years), limit=args.limit)
        report = harvester.run(max_docs=args.docs, target_images=args.target)
    print(report.line())
    return 0 if report.docs else 1


def _years(text: str) -> list[str]:
    out: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(str(y) for y in range(int(a), int(b) + 1))
        elif part:
            out.append(part)
    return out


if __name__ == "__main__":
    raise SystemExit(_main())
