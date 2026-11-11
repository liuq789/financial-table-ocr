"""表格识别后端对比：网格完整性 + 数字召回/精确率（拿 PDF 文字层当参照）。

三份样本 PDF 都带文字层，所以每页的数字是现成的参照答案：把表格单元格里的
数字和该页 PDF 文字里的数字做多重集合比对，**召回**说明漏了多少、**精确**说明
有没有凭空造数字。比"行数看着对不对"客观，而且盯的正是财报最要命的部分。

    # 本地链路（快，不需要服务）
    PYTHONPATH=src $PY tests/bench_tables.py --backend legacy

    # VLM 链路（需要服务：vLLM / SGLang / lmdeploy / 云厂商的 OpenAI 兼容接口）
    PYTHONPATH=src $PY tests/bench_tables.py --backend paddleocr_vl --url http://127.0.0.1:8000/v1

结果 json 落盘到 `work/bench/`，可拿两次的 summary 直接 diff。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config          # noqa: E402
from pipeline import Pipeline           # noqa: E402

NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2,}|\d{4,}")
PCT = re.compile(r"\d+(?:\.\d+)?%")
PDFS = {code: ROOT / "data" / "pdfs" / f"{code}_2023_annual.pdf"
        for code in ("000651", "601187", "000656")}


def grid_stats(table: dict) -> dict:
    cover = [[0] * table["n_cols"] for _ in range(table["n_rows"])]
    for cell in table["cells"]:
        for dr in range(cell["rowspan"]):
            for dc in range(cell["colspan"]):
                r, c = cell["row"] + dr, cell["col"] + dc
                if 0 <= r < table["n_rows"] and 0 <= c < table["n_cols"]:
                    cover[r][c] += 1
    return {"rows": table["n_rows"], "cols": table["n_cols"],
            "cells": len(table["cells"]),
            "holes": sum(1 for row in cover for v in row if v == 0),
            "dup": sum(1 for row in cover for v in row if v > 1),
            "empty": sum(1 for c in table["cells"] if not c["ocr"].strip())}


def tokens(text: str) -> Counter:
    text = text.replace("（", "(").replace("）", ")").replace("−", "-")
    text = re.sub(r"[ \u00a0]", "", text)
    return Counter(NUM.findall(text) + PCT.findall(text))


def page_numbers(stem: str) -> Counter:
    """该页 PDF 文字层里的数字；PDF 不在或没这一页就返回空。"""
    try:
        import pymupdf

        code, page_part, _ = stem.split("_", 2)
        pdf = PDFS.get(code)
        if pdf is None or not pdf.exists():
            return Counter()
        with pymupdf.open(pdf) as doc:
            return tokens(doc[int(page_part.lstrip("p")) - 1].get_text("text"))
    except Exception as exc:                     # noqa: BLE001  参照缺失不该中断评测
        print(f"  ! 取 PDF 文字失败（{stem}）：{type(exc).__name__}: {exc}")
        return Counter()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="表格识别后端评测")
    p.add_argument("--backend", default="legacy", choices=["legacy", "paddleocr_vl"])
    p.add_argument("--url", default=None, help="模型服务的 base_url")
    p.add_argument("--images", type=Path, default=ROOT / "data" / "images")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张")
    p.add_argument("--out", type=Path, default=ROOT / "work" / "bench")
    args = p.parse_args(argv)

    cfg = load_config()
    cfg["tablerec"]["backend"] = args.backend
    if args.url:
        cfg["tablerec"]["paddleocr_vl"]["base_url"] = args.url
    pipe = Pipeline(cfg)
    args.out.mkdir(parents=True, exist_ok=True)

    images = sorted(args.images.rglob("*.png"))
    if args.limit:
        images = images[:args.limit]

    summary = []
    ref_total = hit_total = out_total = 0
    print(f"{'页面':32s} {'表':>3s} {'行列':>9s} {'格':>4s} {'洞':>3s} "
          f"{'召回':>6s} {'精确':>6s} {'耗时':>6s}")
    for image_path in images:
        image = cv2.imread(str(image_path))
        t0 = time.time()
        try:
            page, _ = pipe.process_page(image, 1, do_preprocess=False)
        except Exception as exc:                 # noqa: BLE001  失败要算进评测
            print(f"{image_path.stem:32s}  FAILED {type(exc).__name__}: {exc}")
            summary.append({"image": image_path.stem, "error": f"{type(exc).__name__}: {exc}"})
            continue
        elapsed = round(time.time() - t0, 1)
        tables = [r for r in page["regions"] if r["type"] == "table"]
        stats = [grid_stats(t) for t in tables]

        got = Counter()
        for t in tables:
            for cell in t["cells"]:
                got.update(tokens(cell["ocr"]))
        ref = page_numbers(image_path.stem)
        hit = sum((ref & got).values())
        ref_total += sum(ref.values())
        hit_total += hit
        out_total += sum(got.values())
        recall = hit / max(1, sum(ref.values()))
        precision = hit / max(1, sum(got.values()))

        shape = "+".join(f"{s['rows']}x{s['cols']}" for s in stats) or "-"
        print(f"{image_path.stem:32s} {len(tables):3d} {shape:>9s} "
              f"{sum(s['cells'] for s in stats):4d} {sum(s['holes'] for s in stats):3d} "
              f"{recall:6.1%} {precision:6.1%} {elapsed:5.1f}s")

        summary.append({"image": image_path.stem, "tables": stats,
                        "recall": round(recall, 4), "precision": round(precision, 4),
                        "seconds": elapsed})
        (args.out / f"{args.backend}_{image_path.stem}.json").write_text(
            json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

    recall_all = hit_total / max(1, ref_total)
    precision_all = hit_total / max(1, out_total)
    print(f"\n合计 召回 {recall_all:.1%}  精确 {precision_all:.1%}  "
          f"（参考 {ref_total} / 命中 {hit_total} / 输出 {out_total}）")
    path = args.out / f"_{args.backend}_summary.json"
    path.write_text(json.dumps({"backend": args.backend, "recall": round(recall_all, 4),
                                "precision": round(precision_all, 4),
                                "pages": summary}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
