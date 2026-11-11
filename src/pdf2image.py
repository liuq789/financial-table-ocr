"""主干①：PDF → 页面图片（PyMuPDF）。

    PYTHONPATH=src python -m pdf2image --pdf data/pdfs/xxx.pdf --pages 114-116

扫描件、文字版都能出图；顺带记下每页有没有文字层（有的话后续可以不走 OCR）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pymupdf

from config import ensure_dir, load_config, resolve


def parse_pages(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            pages.extend(range(int(low), int(high) + 1))
        elif part:
            pages.append(int(part))
    return pages


def render_pdf(pdf_path: Path, out_dir: Path, cfg, pages=None,
               dpi: int | None = None) -> list[dict]:
    """逐页渲染成 PNG，返回每页信息。单页失败不影响其它页。"""
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    ensure_dir(out_dir)
    dpi = dpi or int(cfg.render.dpi)

    results: list[dict] = []
    with pymupdf.open(pdf_path) as doc:
        targets = pages or range(1, doc.page_count + 1)
        for page_no in targets:
            if not 1 <= page_no <= doc.page_count:
                print(f"[error] 页码越界：{page_no}/{doc.page_count}", file=sys.stderr)
                continue
            out_path = out_dir / f"p{page_no:03d}.png"
            try:
                page = doc[page_no - 1]
                pix = page.get_pixmap(dpi=dpi)
                pix.save(out_path)
                text = page.get_text("text").strip()
            except Exception as exc:                # noqa: BLE001 单页失败继续
                print(f"[error] {pdf_path.name} p{page_no}：{exc}", file=sys.stderr)
                continue
            results.append({"page_no": page_no, "path": str(out_path),
                            "width": pix.width, "height": pix.height, "dpi": dpi,
                            "page_size_pt": [round(page.rect.width, 1),
                                             round(page.rect.height, 1)],
                            "text_chars": len(text)})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PDF → 页面图片")
    ap.add_argument("--pdf", action="append", required=True)
    ap.add_argument("--pages", default=None, help="如 1-3,7")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    failed = 0
    for pdf in args.pdf:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"[error] 文件不存在：{pdf}", file=sys.stderr)
            failed += 1
            continue
        out_dir = resolve(cfg.paths.work) / "pages" / pdf_path.stem
        started = time.time()
        try:
            results = render_pdf(pdf_path, out_dir, cfg, parse_pages(args.pages), args.dpi)
        except Exception as exc:                    # noqa: BLE001
            print(f"[error] {pdf_path.name}：{exc}", file=sys.stderr)
            failed += 1
            continue
        (out_dir / "_report.json").write_text(
            json.dumps({"pages": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        if results:
            first = results[0]
            print(f"{pdf_path.name} → {len(results)} 页，{first['width']}x{first['height']}，"
                  f"文字层 {sum(1 for r in results if r['text_chars'])}/{len(results)} 页，"
                  f"{time.time() - started:.1f}s → {out_dir}")
        else:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
