"""从 PDF 里挑出"含数据表"的页，并渲成 PNG。

为什么不用 PyMuPDF 的 find_tables：实测在中文财报上不可靠——格力 p114 资产负债表
（4 列 50 行）被它识别成 2 行 2 列。所以这里不走"找表格"，而是走**数据行统计**：

    同一基线上有 >=2 个纯数字 token（且该行总 token >=3）→ 记为一个数据行
    一页里数据行 >= min_rows → 判为表格页

这个信号对有线表、三线表、无线表都成立，因为财报表格的特征是"数字成列"，与线无关。
实测格力年报 249 页里 129 页命中，利润表/资产负债表/现金流量表都排在最前（51/34/41 行）。

候选框 boxes：把命中的数据行按纵向空白（> gap pt）分块，块内取所有词的并集。
它只是"粗定位"，不能当检测真值用。

单独跑：
    PYTHONPATH=src:tests python -m crawl.pagepick --pdf data/pdfs/000651_2023_annual.pdf --limit 6
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pymupdf

from crawl.contracts import PageHit, RenderedPage

log = logging.getLogger(__name__)

# 纯数字：允许千分位、小数、负号、括号负数、百分号、破折号占位
_NUM = re.compile(r"^[（(]?[-−—]?[\d,]+(\.\d+)?[)）]?%?$")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_BASELINE_TOL = 4.0     # pt，同一行的 y 容差
_BLOCK_GAP = 30.0       # pt，数据行之间超过这个空白就分块
_MIN_TOKENS = 3


@dataclass
class _Row:
    y0: float
    y1: float
    x0: float
    x1: float
    nums: int


def _rows_of_page(page: pymupdf.Page) -> list[_Row]:
    """按基线聚词，返回"数据行"。"""
    buckets: dict[int, list[tuple]] = {}
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        buckets.setdefault(round(y0 / _BASELINE_TOL), []).append((x0, y0, x1, y1, word))

    rows: list[_Row] = []
    for words in buckets.values():
        nums = sum(1 for w in words if _NUM.match(w[4]))
        if nums < 2 or len(words) < _MIN_TOKENS:
            continue
        # 至少得有一个中文标签，纯数字行（页码、金额堆）不算
        if not any(_CJK.search(w[4]) for w in words):
            continue
        rows.append(
            _Row(
                y0=min(w[1] for w in words),
                y1=max(w[3] for w in words),
                x0=min(w[0] for w in words),
                x1=max(w[2] for w in words),
                nums=nums,
            )
        )
    rows.sort(key=lambda r: r.y0)
    return rows


def _merge_blocks(rows: list[_Row]) -> list[tuple[float, float, float, float]]:
    """纵向相邻的数据行合成块；块之间留白 > _BLOCK_GAP 就断开。"""
    blocks: list[list[_Row]] = []
    for row in rows:
        if blocks and row.y0 - blocks[-1][-1].y1 <= _BLOCK_GAP:
            blocks[-1].append(row)
        else:
            blocks.append([row])
    return [
        (min(r.x0 for r in b), min(r.y0 for r in b), max(r.x1 for r in b), max(r.y1 for r in b))
        for b in blocks
        if len(b) >= 2          # 单行不算表
    ]


def scan_page(page: pymupdf.Page) -> PageHit:
    """一页打分。纯函数，不改 page。"""
    rows = _rows_of_page(page)
    return PageHit(
        page_no=page.number + 1,
        rows=len(rows),
        max_cols=max((r.nums for r in rows), default=0),
        chars=len(page.get_text("text")),
        boxes=tuple(_merge_blocks(rows)),
    )


def pick_pages(doc: pymupdf.Document, min_rows: int, max_pages: int) -> list[PageHit]:
    """全文扫一遍，按数据行数排序取前 max_pages 页（回到页码顺序）。"""
    hits = [h for h in (scan_page(p) for p in doc) if h.rows >= min_rows]
    hits.sort(key=lambda h: (-h.rows, h.page_no))
    return sorted(hits[:max_pages], key=lambda h: h.page_no)


def render_page(doc: pymupdf.Document, page_no: int, dpi: int) -> RenderedPage:
    """渲一页成 PNG。失败抛 pymupdf 自己的异常，由调用方按"单份失败"处理。"""
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return RenderedPage(
        hit=scan_page(page),
        png=pix.tobytes("png"),
        width=pix.width,
        height=pix.height,
        dpi=dpi,
    )


def extract_pages(pdf_bytes: bytes, min_rows: int, max_pages: int, dpi: int) -> tuple[int, list[RenderedPage]]:
    """字节流 → (总页数, 选中的渲染页)。这就是"一份 PDF 进来、页图出去"的全部。"""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            raise ValueError("PDF 没有页")
        n_pages = doc.page_count
        hits = pick_pages(doc, min_rows=min_rows, max_pages=max_pages)
        pages = [render_page(doc, h.page_no, dpi) for h in hits]
    return n_pages, pages


def _main() -> int:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="看一份 PDF 会被挑出哪些页（不落盘）")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--min-rows", type=int, default=5)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pages = extract_pages(Path(args.pdf).read_bytes(), args.min_rows, args.limit, args.dpi)[1]
    for p in pages:
        h = p.hit
        print(
            f"p{h.page_no:>4}  行={h.rows:>3}  列≈{h.n_cols:>2}  字={h.chars:>5}  "
            f"块={len(h.boxes)}  图={len(p.png) // 1024:>4}KB  {p.width}x{p.height}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
