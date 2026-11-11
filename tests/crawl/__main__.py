"""crawl 的统一入口。

    PYTHONPATH=src:tests python -m crawl status
    PYTHONPATH=src:tests python -m crawl catalog --kind annual --years 2020-2024
    PYTHONPATH=src:tests python -m crawl run --docs 20
    PYTHONPATH=src:tests python -m crawl scan --pdf data/pdfs/000651_2023_annual.pdf
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="crawl", description="财报抓取（巨潮 → PDF → 表格页图）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="台账 + 磁盘")
    p_status.set_defaults(func=_status)

    p_cat = sub.add_parser("catalog", help="只登记公告元数据")
    p_cat.add_argument("--kind", default="annual", choices=["annual", "semiannual"])
    p_cat.add_argument("--years", default="2023", help="披露年份，如 2020-2024 或 2021,2023")
    p_cat.add_argument("--limit", type=int, default=0, help="每年最多登记几条")
    p_cat.set_defaults(func=_catalog)

    p_run = sub.add_parser("run", help="批量下载 + 出图")
    p_run.add_argument("--docs", type=int, default=0)
    p_run.add_argument("--target", type=int, default=None)
    p_run.set_defaults(func=_run)

    p_scan = sub.add_parser("scan", help="看一份 PDF 会挑出哪些页")
    p_scan.add_argument("--pdf", required=True)
    p_scan.add_argument("--min-rows", type=int, default=5)
    p_scan.add_argument("--limit", type=int, default=8)
    p_scan.set_defaults(func=_scan)

    args = ap.parse_args(argv)
    return args.func(args)


def _ctx():
    from config import load_config, resolve
    from crawl.cninfo import CninfoClient
    from crawl.runner import Harvester
    from crawl.settings import build_settings
    from crawl.store import Store
    from logging_setup import setup_logging

    setup_logging()
    cfg = build_settings(load_config(), resolve)
    store = Store(cfg).open()
    return cfg, store, Harvester(cfg, store, CninfoClient(cfg))


def _status(_args) -> int:
    import json

    cfg, store, _h = _ctx()
    try:
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        print(f"磁盘剩余 {store.free_gb():.1f} GB / 数据目录 {cfg.data_root}")
    finally:
        store.close()
    return 0


def _catalog(args) -> int:
    from crawl.runner import _years

    _cfg, store, harvester = _ctx()
    try:
        added = harvester.catalog(args.kind, _years(args.years), limit=args.limit)
        print(f"新增 {added} 条待办")
    finally:
        store.close()
    return 0


def _run(args) -> int:
    _cfg, store, harvester = _ctx()
    try:
        report = harvester.run(max_docs=args.docs, target_images=args.target)
        print(report.line())
    finally:
        store.close()
    return 0 if report.docs else 1


def _scan(args) -> int:
    from pathlib import Path

    from crawl.pagepick import extract_pages

    pages = extract_pages(Path(args.pdf).read_bytes(), args.min_rows, args.limit, 200)[1]
    for p in pages:
        h = p.hit
        print(f"p{h.page_no:>4} 行={h.rows:>3} 列≈{h.n_cols:>2} 块={len(h.boxes)} "
              f"图={len(p.png) // 1024:>4}KB {p.width}x{p.height}")
    print(f"共 {len(pages)} 页")
    return 0


if __name__ == "__main__":
    sys.exit(main())
