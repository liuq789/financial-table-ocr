"""抓取链路用例：标题过滤 / 筛页 / 台账。不联网，只用 data/pdfs 里的真实样本。

抓取链路不在主干（`src/`），整个包放在 `tests/crawl/`，所以要同时把两个目录加进路径：

    PYTHONPATH=src:tests $PY tests/check_crawl.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "tests"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from crawl.cninfo import is_report, to_announcement          # noqa: E402
from crawl.contracts import Announcement, CrawlSettings      # noqa: E402
from crawl.pagepick import extract_pages                     # noqa: E402
from crawl.store import Store                                # noqa: E402

SAMPLE = ROOT / "data" / "pdfs" / "000651_2023_annual.pdf"
_fail: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        _fail.append(name)


def case_titles() -> None:
    keep = ["2023年年度报告", "2023年年度报告全文（更新后）", "2022 年半年度报告"]
    drop = ["2023年年度报告摘要", "2023年年度报告（英文版）", "关于2023年年度报告的更正公告",
            "2023年度审计报告", "2023年社会责任报告"]
    for t in keep:
        kind = "semiannual" if "半年" in t else "annual"
        check(f"收 {t}", is_report(t, kind))
    for t in drop:
        kind = "semiannual" if "半年" in t else "annual"
        check(f"弃 {t}", not is_report(t, kind))

    raw = {
        "secCode": "000651", "secName": "格力电器", "announcementTitle": "2023年年度报告",
        "adjunctUrl": "finalpage/2024-04-30/1219999999.PDF", "adjunctSize": 3800,
        "announcementTime": 1714406400000,
    }
    ann = to_announcement(raw, "annual")
    check("转契约", ann is not None and ann.key == "000651_2023_annual" and ann.year == "2023")
    check("直链", ann is not None and ann.url.startswith("http://static.cninfo.com.cn/finalpage/"))
    check("拒非 PDF", to_announcement({**raw, "adjunctUrl": "a.html"}, "annual") is None)


def case_pages() -> None:
    if not SAMPLE.exists():
        check("样本存在", False, str(SAMPLE))
        return
    n_pages, pages = extract_pages(SAMPLE.read_bytes(), min_rows=5, max_pages=8, dpi=150)
    nums = [p.hit.page_no for p in pages]
    check("页数读出", n_pages > 200, str(n_pages))
    check("挑出 8 页", len(pages) == 8, str(len(pages)))
    # 三大报表 + 资产构成是已知表格页
    check("命中利润表 p116", 116 in nums, str(nums))
    check("页图非空", all(len(p.png) > 20_000 for p in pages))
    check("尺寸合理", all(p.width > 800 for p in pages))
    check("候选框不越界", all(
        b[0] >= -1 and b[1] >= -1 and b[2] <= p.hit.bbox[2] + 1 for p in pages for b in p.hit.boxes
    ) if pages else False)
    # min_rows 抬高会减少页数
    fewer = extract_pages(SAMPLE.read_bytes(), min_rows=40, max_pages=8, dpi=72)[1]
    check("阈值越高留页越少", len(fewer) < len(pages), f"{len(fewer)} vs {len(pages)}")


def case_store() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="crawl_"))
    try:
        cfg = CrawlSettings(data_root=tmp)
        ann = Announcement("000651", "格力电器", "2023年年度报告", "2023", "annual",
                           "http://static.cninfo.com.cn/x.PDF", 3800, "2024-04-30")
        with Store(cfg) as store:
            check("入账", store.add(ann) is True)
            check("重复入账不重复", store.add(ann) is False)
            check("待办可取", len(store.pending(10)) == 1)
            n_pages, pages = extract_pages(SAMPLE.read_bytes(), 5, 2, dpi=72)
            store.mark_running(ann.key)
            store.save_pdf(ann, b"%PDF-1.7 test")
            written = store.save_pages(ann, n_pages, pages)
            check("页图落盘", written == 2 == len(list(store.page_dir(ann).glob("*.png"))))
            check("index.json 有", (store.page_dir(ann) / "index.json").exists())
            store.mark_done(__import__("crawl.contracts", fromlist=["DocOutcome"]).DocOutcome(
                ann=ann, n_pages=n_pages, pages=pages, kept_pdf=True, pdf_bytes=12))
            st = store.stats()
            check("台账统计", st["by_status"]["done"]["docs"] == 1 and st["images"] == 2, str(st))
            check("待办清空", store.pending(10) == [])
            store.mark_failed("000651_2023_annual", "boom")
            check("失败可重试", len(store.pending(10)) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    for case in (case_titles, case_pages, case_store):
        print(case.__name__)
        case()
    print(f"\n{'全部通过' if not _fail else '失败：' + ', '.join(_fail)}")
    return 0 if not _fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
