"""整份文档 → json + excel。一页按「分区」输出。

一条链路：
    PDF/图片 → 页图(PyMuPDF) → 矫正(倾斜+180) → 版面找表格(PP-DocLayoutV3)
      → [表格识别] → 表格外正文 OCR → 按分区输出 → json / xlsx

[表格识别] 有两条路，由 config 的 `tablerec.backend` 选，互不影响：

    legacy  本地 ONNX：分类(有线/无线) → 结构识别(unet / 读光) → PP-OCR 逐格识别
    paddleocr_vl  表格裁图整块送 PaddleOCR-VL（本机 vLLM 的 OpenAI 兼容接口），
                  结构与文字一次出（本地结构模型不参与；单张表失败按
                  `tablerec.fallback_legacy` 退回 legacy）

版面找表两条路都用本地的 PP-DocLayoutV3，只换"表内怎么读"。

输出约定
--------
一页拆成若干**分区**，按阅读顺序（先上下、再左右）排。分区只有两种：

    table  表格。内容是行列网格里的单元格（cell 带 row/col/rowspan/colspan/ocr）
    char   文字块。内容是若干文字行（line 带 text/bbox）

json 每页结构::

    {"page": 1, "width": W, "height": H,
     "regions": [
        {"type": "char", "bbox": [x0, y0, x1, y1],
         "text": "六、资产及负债状况分析",
         "lines": [{"text": "六、资产及负债状况分析", "bbox": [...]}]},
        {"type": "table", "bbox": [...], "kind": "wired", "kind_score": 0.99,
         "n_rows": 15, "n_cols": 7,
         "cells": [{"row": 0, "col": 0, "rowspan": 1, "colspan": 7,
                    "bbox": [...], "ocr": "项目"}]}
     ]}

excel 每页一个 sheet：按分区的上下位置铺成一张大表，**表格区域画粗框线并还原
合并单元格，文字区域不加任何框线**。

用法：
    PYTHONPATH=src python -m pipeline --input data/pdfs/000651_2023_annual.pdf --pages 114-116
    PYTHONPATH=src python -m pipeline --input data/images/ruled/000651_p029_asset_change.png
    PYTHONPATH=src python -m pipeline --input xxx.pdf --backend paddleocr_vl \
        --model-url http://127.0.0.1:8000/v1
    PYTHONPATH=src python -m web.server --port 8010      # 网页：上传即出 json + excel
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]      # financial_ocr/
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from export import write_excel, write_json                      # noqa: E402
from resource_guard import Budget, Limits, ResourceShortage     # noqa: E402
from logging_setup import TOTAL_ORDER, Stopwatch                 # noqa: E402
from model.layout import Layout                                  # noqa: E402
from model.ppocr import Ppocr                                    # noqa: E402
from model.table_cls import TableKind                            # noqa: E402
from model.table_structure import build as build_struct          # noqa: E402
from model.paddleocr_vl import PaddleOcrVl, PaddleOcrVlError              # noqa: E402
from preprocess import correct_flip, correct_skew, imread        # noqa: E402

logger = logging.getLogger(__name__)
BACKENDS = ("legacy", "paddleocr_vl")


def _cell_dict(cell, ox: int = 0, oy: int = 0) -> dict:
    """把有线(unet.Cell) / 无线(lore.LoreCell) 统一成一种结构（页坐标）。"""
    if hasattr(cell, "poly"):                       # LoreCell：只有起止行列
        pts = np.asarray(cell.poly)
        bbox = (int(pts[:, 0].min()) + ox, int(pts[:, 1].min()) + oy,
                int(pts[:, 0].max()) + ox, int(pts[:, 1].max()) + oy)
        row, col = cell.row_start, cell.col_start
        rowspan = cell.row_end - cell.row_start + 1
        colspan = cell.col_end - cell.col_start + 1
    else:                                           # unet.Cell
        bbox = (int(cell.bbox[0]) + ox, int(cell.bbox[1]) + oy,
                int(cell.bbox[2]) + ox, int(cell.bbox[3]) + oy)
        row, rowspan, col, colspan = cell.r, cell.rowspan, cell.c, cell.colspan
    return {"row": int(row), "col": int(col),
            "rowspan": int(max(1, rowspan)), "colspan": int(max(1, colspan)),
            "bbox": list(bbox), "ocr": ""}


def _crop(image: np.ndarray, bbox, pad: int = 2) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 + pad), max(0, y0 + pad)
    x1, y1 = min(w, x1 - pad), min(h, y1 - pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((0, 0, 3), np.uint8)
    return image[y0:y1, x0:x1]


def _count_rules(profile: np.ndarray, thresh: float) -> int:
    """把线剖面里超过阈值的位置数成"几条线"（相邻的算同一条）。"""
    idx = np.flatnonzero(profile > thresh)
    count, prev = 0, -10
    for i in idx:
        if i - prev > 3:
            count += 1
        prev = i
    return count


def rule_counts(crop: np.ndarray) -> tuple[int, int]:
    """数一数长横线和长竖线各几条。框线表的竖线是判别有线的强证据。

    二值化阈值要用 200：财报的表格线是浅灰细线，用 160 会整条漏掉
    （p029 实测 160 → 0 横线，200 → 13 横线）。
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw = (gray < 200).astype(np.uint8)
    h, w = bw.shape[:2]
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             np.ones((1, max(20, w // 4)), np.uint8))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            np.ones((max(20, h // 4), 1), np.uint8))
    return (_count_rules(horiz.sum(axis=1), w * 0.4),
            _count_rules(vert.sum(axis=0), h * 0.4))


def looks_wired(crop: np.ndarray) -> bool:
    """有完整网格（横线够多 + 有**内部**竖线）就肯定是有线表。

    分类器会在版式相近的表上判错——实测 p114 的框线资产负债表给了
    wireless 0.916，结果走了读光，读光只返回有字的格子，整张表到处是空洞。
    所以竖线是强证据时直接覆盖分类器。

    竖线要求 ≥3 条：三线表只有左右两条边框竖线（实测 p162 是 9 横 2 竖），
    那种表 unet 也分不出列，不能算有线。
    """
    rows, cols = rule_counts(crop)
    return rows >= 3 and cols >= 3


def _expand(rect, dx_ratio: float = 0.02, dy_ratio: float = 0.10,
            min_px: int = 4) -> list[int]:
    """检测框比字形紧，横向按比例外扩，否则数字首尾会被切。"""
    x0, y0, x1, y1 = rect
    dx = max(min_px, int((x1 - x0) * dx_ratio))
    dy = max(2, int((y1 - y0) * dy_ratio))
    return [x0 - dx, y0 - dy, x1 + dx, y1 + dy]


def _line_in_cell(cell_bbox, line, pad: int = 3) -> list[int]:
    """单元格里的一行：y 只取这一行，x 取到格子边界。

    检测框对数字比字形紧，首尾会被切（"157,484,332,251.39" 变成
    "157,484,332,25"），所以横向一律铺满格子。
    """
    dy = max(2, int((line[3] - line[1]) * 0.12))
    return [cell_bbox[0] + pad, line[1] - dy, cell_bbox[2] - pad, line[3] + dy]


def _overlap(a, b) -> float:
    """两个矩形的交集面积。"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(max(0, w) * max(0, h))


def _inside(box, rect, tol: int = 6) -> bool:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return (rect[0] - tol <= cx <= rect[2] + tol) and (rect[1] - tol <= cy <= rect[3] + tol)


def _iou(a, b) -> float:
    """交并比，用来判断两个框是不是同一张表。"""
    inter = _overlap(a, b)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _count_marks(values: list[int], tol: int) -> int:
    """一维坐标聚成"几行/几列"：相邻差不超 tol 的算同一处。"""
    count, anchor = 0, None
    for value in sorted(values):
        if anchor is None or value - anchor > tol:
            count += 1
            anchor = value
    return count


def looks_like_table(rect, lines, min_rows: int, min_cols: int) -> bool:
    """用文字排布判断这块区域像不像表格：行数够多、左边缘够多。

    版面模型对"又宽又扁 + 大量留白"的表会掉到 0.37~0.39，但这类区域的文字
    有特征：每行几个数字、左侧科目对齐。整段正文只有 1 个左边缘（两端对齐），
    多栏正文只有 2 个，都会落选；签名区虽然有多列但只有 2 行，也会落选。
    """
    inside = [b for b in lines if _inside(b, rect)]
    if len(inside) < min_rows:
        return False
    heights = sorted(b[3] - b[1] for b in inside)
    tol_y = max(6, int(heights[len(heights) // 2] * 0.6))
    rows = _count_marks([(b[1] + b[3]) // 2 for b in inside], tol_y)
    tol_x = max(8, int((rect[2] - rect[0]) * 0.02))
    cols = _count_marks([b[0] for b in inside], tol_x)
    return rows >= min_rows and cols >= min_cols


def _dedup(rects: list[list[int]], thr: float = 0.7) -> list[list[int]]:
    """去掉重复检测：同一段文字被检出两次会让两个格子拿到同样的文本。"""
    out: list[list[int]] = []
    for b in sorted(rects, key=lambda r: -(r[2] - r[0]) * (r[3] - r[1])):
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        if all(_overlap(b, o) / area < thr for o in out):
            out.append(b)
    return out


def _group_lines(items: list[dict], gap_ratio: float = 1.5) -> list[list[dict]]:
    """把非表格的文字行聚成块：垂直间距小于行高 1.5 倍且水平有重叠就算同一块。"""
    blocks: list[list[dict]] = []
    boxes: list[list[int]] = []
    for item in sorted(items, key=lambda t: (t["bbox"][1], t["bbox"][0])):
        line = item["bbox"]
        height = max(1, line[3] - line[1])
        for i in range(len(blocks) - 1, -1, -1):
            box = boxes[i]
            vgap = line[1] - box[3]
            overlap = min(box[2], line[2]) - max(box[0], line[0])
            if vgap < max(10, height * gap_ratio) and overlap > 0:
                blocks[i].append(item)
                boxes[i] = [min(box[0], line[0]), min(box[1], line[1]),
                            max(box[2], line[2]), max(box[3], line[3])]
                break
        else:
            blocks.append([item])
            boxes.append(list(line))
    return blocks


def _axis_spans(cells: list[dict], index_key: str, span_key: str,
                n: int, axis: int) -> dict[int, tuple[int, int]]:
    """从已知单元格反推每一行/列的像素范围（MinerU 那套"由格子反推网格"）。

    一个跨列的单元格只对首列贡献左边界、对末列贡献右边界，中间列不参与，
    否则会把整跨的宽度摊到每个子列上。
    """
    lo_at: dict[int, int] = {}
    hi_at: dict[int, int] = {}
    for cell in cells:
        start = cell[index_key]
        end = start + cell[span_key] - 1
        lo, hi = cell["bbox"][axis * 2], cell["bbox"][axis * 2 + 1]
        lo_at[start] = min(lo_at.get(start, lo), lo)
        hi_at[end] = max(hi_at.get(end, hi), hi)

    out: dict[int, tuple[int, int]] = {}
    for i in range(n):
        lo, hi = lo_at.get(i), hi_at.get(i)
        if lo is None and hi is None:
            continue
        if hi is None:                                # 只有左边界：借下一列的左边界
            hi = lo_at.get(i + 1, lo)
        if lo is None:                                # 只有右边界：借上一列的右边界
            lo = hi_at.get(i - 1, hi)
        out[i] = (lo, hi)

    if not out:
        return out
    known = sorted(out)
    for i in range(n):                               # 整行/整列都缺时借邻居
        if i not in out:
            left = max((k for k in known if k < i), default=None)
            right = min((k for k in known if k > i), default=None)
            out[i] = out[left] if left is not None else out[right]
    return out


def drop_overlaps(cells: list[dict]) -> list[dict]:
    """去掉互相压住的单元格：先放大格子，起点已被占的后来者直接丢。

    无线模型的输出偶尔会有两个格子圈到同一块区域，不处理的话 excel 的
    merge 会冲突，导出的表格也会错位。
    """
    occupied: set[tuple[int, int]] = set()
    kept: list[dict] = []
    for cell in sorted(cells, key=lambda c: (-(c["rowspan"] * c["colspan"]),
                                             c["row"], c["col"])):
        if (cell["row"], cell["col"]) in occupied:
            continue
        kept.append(cell)
        for dr in range(cell["rowspan"]):
            for dc in range(cell["colspan"]):
                occupied.add((cell["row"] + dr, cell["col"] + dc))
    kept.sort(key=lambda c: (c["row"], c["col"]))
    return kept


def fill_grid(cells: list[dict], n_rows: int, n_cols: int) -> list[dict]:
    """把 n_rows×n_cols 里没被任何单元格覆盖的位置补成空格子。

    无线表格模型只返回"有文字的格子"，空单元格一个都不给，导出的表格因此
    到处缺口。这里按行列反推出空洞的像素范围补上，保证网格始终完整。
    """
    if n_rows <= 0 or n_cols <= 0:
        return cells

    covered = set()
    for cell in cells:
        for dr in range(cell["rowspan"]):
            for dc in range(cell["colspan"]):
                covered.add((cell["row"] + dr, cell["col"] + dc))
    holes = [(r, c) for r in range(n_rows) for c in range(n_cols)
             if (r, c) not in covered]
    if not holes:
        return cells

    rows_y = _axis_spans(cells, "row", "rowspan", n_rows, axis=1)
    cols_x = _axis_spans(cells, "col", "colspan", n_cols, axis=0)
    filled = list(cells)
    for row, col in holes:
        y0, y1 = rows_y.get(row, (0, 0))
        x0, x1 = cols_x.get(col, (0, 0))
        filled.append({"row": row, "col": col, "rowspan": 1, "colspan": 1,
                       "bbox": [int(x0), int(y0), int(x1), int(y1)], "ocr": ""})
    filled.sort(key=lambda c: (c["row"], c["col"]))
    return filled


def _trim_empty_rows(cells: list[dict], n_rows: int) -> tuple[list[dict], int]:
    """裁掉大表格首尾那些一个格子都没文字的行（模型在外框处多推的空行）。"""
    filled = {c["row"] for c in cells if c["ocr"]}
    if not filled:
        return cells, n_rows
    first, last = min(filled), max(filled)
    if first == 0 and last == n_rows - 1:
        return cells, n_rows
    out = []
    for cell in cells:
        r0 = max(cell["row"], first)
        r1 = min(cell["row"] + cell["rowspan"] - 1, last)
        if r1 < r0:
            continue
        out.append({**cell, "row": r0 - first, "rowspan": r1 - r0 + 1})
    return out, last - first + 1


class Pipeline:
    def __init__(self, cfg) -> None:
        base = ROOT / "models"
        self.cfg = cfg
        if "tablerec" not in cfg:
            raise ValueError("config.yaml 缺少 tablerec 段（backend: legacy|paddleocr_vl）")
        self.backend = str(cfg["tablerec"]["backend"]).strip().lower()
        if self.backend not in BACKENDS:
            raise ValueError(f"tablerec.backend 只能是 {BACKENDS}，当前 {self.backend!r}")
        self.ocr = Ppocr(cfg.textdet)
        self.layout = Layout(base / cfg["layout"]["model"],
                             float(cfg["layout"]["score_thresh"]))
        self.margin = int(cfg["layout"]["margin"])
        self.recover_thresh = float(cfg["layout"].get("recover_thresh", 1.1))
        self.recover_min_lines = int(cfg["layout"].get("recover_min_lines", 5))
        self.recover_min_cols = int(cfg["layout"].get("recover_min_cols", 3))
        self.table_model = (PaddleOcrVl(cfg["tablerec"]["paddleocr_vl"])
                            if self.backend == "paddleocr_vl" else None)
        self.fallback_legacy = bool(cfg["tablerec"].get("fallback_legacy", True))
        self._kind: TableKind | None = None          # legacy 专用，按需加载
        self._struct = {}

    def _kind_model(self) -> TableKind:
        if self._kind is None:
            self._kind = TableKind(ROOT / "models" / "table" / "table_cls.onnx")
        return self._kind

    def _model(self, kind: str):
        if kind not in self._struct:                 # 结构模型很重，按需加载一次
            self._struct[kind] = build_struct(kind, self.cfg)
        return self._struct[kind]

    def _table_rects(self, image: np.ndarray, lines: list[list[int]]
                     ) -> list[tuple[int, int, int, int]]:
        """版面给的表框 + 低分候选的几何复核（解决宽扁表被阈值砍掉）。"""
        accepted: list[tuple[tuple[int, int, int, int], float]] = []
        pending: list[tuple[tuple[int, int, int, int], float]] = []
        for rect, score in self.layout.detect_candidates(image):
            if score >= self.layout.score_thresh:
                accepted.append((rect, score))
            elif score >= self.recover_thresh:
                pending.append((rect, score))

        for rect, score in pending:
            if any(_iou(rect, kept) > 0.3 for kept, _ in accepted):
                continue                             # 已经框住同一张表，别重复
            if looks_like_table(rect, lines, self.recover_min_lines, self.recover_min_cols):
                logger.info("版面低分框 %.2f 通过文字复核，采纳为表格：%s",
                            score, list(rect))
                accepted.append((rect, score))
            else:
                logger.debug("版面低分框 %.2f 文字复核未通过，丢弃：%s", score, list(rect))

        rects = [rect for rect, _ in accepted]
        rects.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        return rects

    # ------------------------------------------------------------ 表格：选路
    def _read_table(self, crop: np.ndarray, page_no: int, bbox, lines,
                    sw: Stopwatch, fallbacks: list[dict]) -> dict:
        """一张表的识别。模型侧失败时按配置退回 legacy，并把原因记进 fallbacks。"""
        with sw.span("table"):
            if self.table_model is None:
                return self._legacy_table(crop, bbox, lines)

            x0, y0, x1, y1 = bbox
            logger.debug("表格裁图 %s 尺寸=%dx%d", list(bbox), x1 - x0, y1 - y0)
            try:
                with sw.span("model"):
                    result = self.table_model.predict(crop)
                return self._table_from_model(result, bbox)
            except PaddleOcrVlError as exc:
                if not self.fallback_legacy:             # 明确要求纯模型就直接炸
                    raise
                logger.warning("p%d 表格 %s 走 PaddleOCR-VL 失败，退回本地链路：%s",
                               page_no, list(bbox), exc)
                fallbacks.append({"page": page_no, "bbox": list(bbox),
                                  "reason": str(exc)[:300]})
                return self._legacy_table(crop, bbox, lines)

    @staticmethod
    def _table_from_model(result, bbox) -> dict:
        """模型结果 → 分区里的 table 结构。bbox 是裁图内坐标，加表格左上角偏移。"""
        x0, y0 = bbox[0], bbox[1]
        cells = [{"row": c.row, "col": c.col,
                  "rowspan": c.rowspan, "colspan": c.colspan,
                  "bbox": [c.bbox[0] + x0, c.bbox[1] + y0,
                           c.bbox[2] + x0, c.bbox[3] + y0],
                  "ocr": c.text} for c in result.cells]
        return {"bbox": list(bbox), "kind": "paddleocr_vl", "kind_score": 0.0,
                "n_rows": result.n_rows, "n_cols": result.n_cols,
                "cells": cells, "_lines": None}      # _lines=None 表示不再逐格 OCR

    def _legacy_table(self, crop: np.ndarray, bbox, lines) -> dict:
        x0, y0, x1, y1 = bbox
        kind, kind_score = self._kind_model().predict(crop)
        # 有完整网格（横线≥3 且竖线≥2）就直接算有线表：分类器在财报上不可靠
        # （p114 框线资产负债表给 wireless 0.916），判错的代价是走读光、整表开洞
        if looks_wired(crop):
            kind = "wired"
        elif kind_score < 0.8:
            kind = "wireless"
        out = self._model(kind).predict(crop)
        raw = out.cells if hasattr(out, "cells") else out     # unet 返 TableStruct
        cells = [_cell_dict(c, x0, y0) for c in raw]

        # 文本行归属：按与单元格的交叠面积取最大，避免一行落进多个格子
        per_cell: list[list[list[int]]] = [[] for _ in cells]
        for line in lines:
            area = max(1.0, (line[2] - line[0]) * (line[3] - line[1]))
            best, best_ov = -1, 0.0
            for ci, cell in enumerate(cells):
                ov = _overlap(line, cell["bbox"])
                if ov > best_ov:
                    best, best_ov = ci, ov
            if best >= 0 and best_ov / area > 0.5:
                per_cell[best].append(line)
        for group in per_cell:
            group.sort(key=lambda b: ((b[1] + b[3]) // 12, b[0]))

        n_rows = max((c["row"] + c["rowspan"] for c in cells), default=0)
        n_cols = max((c["col"] + c["colspan"] for c in cells), default=0)
        return {"bbox": [x0, y0, x1, y1], "kind": kind,
                "kind_score": round(kind_score, 4),
                "n_rows": n_rows, "n_cols": n_cols,
                "cells": cells, "_lines": per_cell}

    # ---------------------------------------------------------------- 单页
    def process_page(self, image: np.ndarray, page_no: int, do_preprocess: bool
                     ) -> tuple[dict, np.ndarray]:
        sw = Stopwatch()
        t_page = time.perf_counter()

        with sw.span("detect"):
            boxes = self.ocr.detect(image)
        skew = flip = None
        if do_preprocess:
            with sw.span("preprocess"):
                image, skew, boxes = correct_skew(image, self.ocr,
                                                  self.cfg.preprocess.angle, boxes)
                image, flip, boxes = correct_flip(image, self.ocr,
                                                  self.cfg.preprocess.flip, boxes)

        # 整页检测框转成 (x0,y0,x1,y1)：单元格内按行归组，表格外的算 ocr
        lines = []
        for box in boxes:
            xs = [p[0] for p in box.points]
            ys = [p[1] for p in box.points]
            lines.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
        lines = _dedup(lines)

        with sw.span("layout"):
            table_rects = self._table_rects(image, lines)
        h, w = image.shape[:2]
        tables = []
        fallbacks: list[dict] = []      # 模型不可用时退回本地链路的记录
        for rect in table_rects:
            x0 = max(0, rect[0] - self.margin)
            y0 = max(0, rect[1] - self.margin)
            x1 = min(w, rect[2] + self.margin)
            y1 = min(h, rect[3] + self.margin)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            tables.append(self._read_table(crop, page_no, (x0, y0, x1, y1),
                                           lines, sw, fallbacks))

        # 分区内文本 + 表格外文本，统一一次批量识别
        jobs: list[np.ndarray] = []
        refs: list[tuple[str, int, int]] = []
        ocr_lines: list[dict] = []
        for ti, table in enumerate(tables):
            if table["_lines"] is None:              # 模型已出文字，不再逐格 OCR
                continue
            for ci, cell in enumerate(table["cells"]):
                group = table["_lines"][ci]
                if group:                            # 沿检测框逐行取 y，x 铺满格子
                    for b in group:
                        patch = _crop(image, _line_in_cell(cell["bbox"], b))
                        if patch.size:
                            jobs.append(patch)
                            refs.append(("cell", ti, ci))
                else:                                # 格子里没有检测框：整格兜底
                    patch = _crop(image, cell["bbox"], pad=3)
                    if patch.size:
                        jobs.append(patch)
                        refs.append(("cell", ti, ci))
        for b in lines:
            if any(_inside(b, r) for r in table_rects):
                continue
            patch = _crop(image, _expand(b), pad=1)
            if patch.size:
                jobs.append(patch)
                refs.append(("text", -1, len(ocr_lines)))
                ocr_lines.append({"bbox": b, "text": ""})

        with sw.span("ocr"):
            results = self.ocr.recognize(jobs) if jobs else []
        for (tag, ti, ci), text in zip(refs, results):
            text = text.strip()
            if not text:
                continue
            if tag == "cell":
                cell = tables[ti]["cells"][ci]
                # 一个格子里有多行时用换行分隔，不要粘成一串（行线漏检时靠它辨认）
                cell["ocr"] = f"{cell['ocr']}\n{text}" if cell["ocr"] else text
            else:
                ocr_lines[ci]["text"] = text

        with sw.span("post"):
            for table in tables:
                table.pop("_lines", None)
                # 裁掉分区首尾那种一个格子都没文字的行（模型在外框处多推的）
                table["cells"], table["n_rows"] = _trim_empty_rows(
                    table["cells"], table["n_rows"])
                # 先消重（模型偶尔把两个格子圈到同一块），再补齐空格子：
                # 无线模型只返回有字的格子，不补就到处是缺口
                table["cells"] = fill_grid(drop_overlaps(table["cells"]),
                                           table["n_rows"], table["n_cols"])

        # 按分区输出：表格一块，连续的文字行聚成一块
        regions: list[dict] = []
        for table in tables:
            regions.append({"type": "table", "bbox": table["bbox"],
                            "kind": table["kind"], "kind_score": table["kind_score"],
                            "n_rows": table["n_rows"], "n_cols": table["n_cols"],
                            "cells": table["cells"]})
        for block in _group_lines([t for t in ocr_lines if t["text"]]):
            bbox = [min(t["bbox"][0] for t in block), min(t["bbox"][1] for t in block),
                    max(t["bbox"][2] for t in block), max(t["bbox"][3] for t in block)]
            regions.append({"type": "char", "bbox": bbox,
                            "text": "\n".join(t["text"] for t in block),
                            "lines": [{"text": t["text"], "bbox": t["bbox"]} for t in block]})
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        timings = sw.as_dict()
        timings["total"] = int(round((time.perf_counter() - t_page) * 1000))
        page = {"page": page_no, "width": w, "height": h,
                "skew": round(float(skew["total"]), 3) if skew else 0.0,
                "flipped": bool(flip["flipped"]) if flip else False,
                "regions": regions, "timings": timings, "fallbacks": fallbacks}
        n_cell = sum(len(r["cells"]) for r in regions if r["type"] == "table")
        logger.info("p%03d 分区=%d 表=%d 文字=%d 格=%d | %s",
                    page_no, len(regions), len(tables),
                    len(regions) - len(tables), n_cell, sw.line(TOTAL_ORDER))
        return page, image


def _fit_page(image: np.ndarray, max_pixels: int, page_no: int) -> np.ndarray:
    """超过像素上限的页先等比缩小：保住进程，代价是这一页的字会小一点。"""
    if image.size == 0:
        return image
    h, w = image.shape[:2]
    pixels = h * w
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    logger.warning("p%d 分辨率过高（%dx%d=%d 像素），缩到 %dx%d 再识别",
                   page_no, w, h, pixels, size[0], size[1])
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def process(input_path: Path, out_dir: Path, cfg, pages=None, dpi=None,
            do_preprocess: bool = True, pipe: "Pipeline | None" = None,
            budget: "Budget | None" = None) -> dict:
    """整份文档 → json + xlsx。

    `budget` 是资源配额：每页开工前复查一次内存/自身占用，超了就中止这一单并抛
    ResourceShortage——宁可失败一个请求，也不能让进程被 OOM 杀掉。
    """
    input_path, out_dir = Path(input_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = Limits.from_config(cfg)
    sw = Stopwatch()
    t0 = time.perf_counter()

    if input_path.suffix.lower() == ".pdf":
        from pdf2image import render_pdf

        with sw.span("render"):
            rendered = render_pdf(input_path, out_dir / "pages", cfg, pages, dpi)
            # 只留路径，逐页读：60 页 200dpi 的图全读进来要 700MB+，WSL 会被 OOM 杀掉
            items = [(r["page_no"], Path(r["path"])) for r in rendered]
        if len(items) > limits.max_pages:
            raise ValueError(f"这次要跑 {len(items)} 页，超过上限 {limits.max_pages} 页；"
                             f"请用页范围分批跑（config.yaml 的 guard.max_pages 可调）")
        logger.info("渲染 %d 页（dpi=%s）", len(items), dpi or cfg["render"]["dpi"])
    else:
        items = [(1, input_path)]

    pipe = pipe or Pipeline(cfg)                 # 长驻进程（web）可复用已加载的模型
    doc = {"doc": input_path.name, "pages": []}
    for page_no, path in items:
        if budget is not None:
            budget.check(f"第 {page_no} 页开工前")
        image = _fit_page(imread(path), limits.max_page_pixels, page_no)
        try:
            page, _ = pipe.process_page(image, page_no, do_preprocess)
        finally:
            del image                            # 单页峰值也要尽快还给系统
        page["elapsed_s"] = round(page["timings"]["total"] / 1000.0, 2)
        doc["pages"].append(page)

    # 汇总字段先算好再导出：JSON 落盘的内容必须和接口返回的一致
    # （engine / fallbacks / timings 都是给使用方看的元信息，缺了很难判断这批结果是怎么来的）
    totals = Stopwatch()
    for page in doc["pages"]:
        for key, ms in page["timings"].items():
            if key != "total":
                totals.add(key, ms)
    doc["engine"] = pipe.backend
    doc["fallbacks"] = [item for page in doc["pages"]
                        for item in page.get("fallbacks", [])]
    if doc["fallbacks"]:
        logger.warning("%d 张表因模型不可用退回了本地链路（结果质量会差一档）",
                       len(doc["fallbacks"]))

    stem = input_path.stem
    # 先写 Excel，再算总耗时，最后写一次 JSON：保证 JSON 里的 engine/timings
    # 与接口返回完全一致，而且只落盘一次。
    with sw.span("export"):
        write_excel(doc, out_dir / f"{stem}.xlsx")

    elapsed = time.perf_counter() - t0
    doc["elapsed_s"] = round(elapsed, 2)
    doc["timings"] = {**totals.as_dict(), "render": sw.as_dict().get("render", 0),
                      "export": sw.as_dict().get("export", 0),
                      "total": int(round(elapsed * 1000))}
    write_json(doc, out_dir / f"{stem}.json")

    n_tables = sum(1 for p in doc["pages"] for r in p["regions"] if r["type"] == "table")
    n_cells = sum(len(r["cells"]) for p in doc["pages"]
                  for r in p["regions"] if r["type"] == "table")
    logger.info("完成 %s 引擎=%s 页=%d 表=%d 格=%d 总耗时=%.2fs | %s",
                input_path.name, doc["engine"], len(doc["pages"]), n_tables, n_cells, elapsed,
                totals.line(TOTAL_ORDER))
    logger.info("产物 %s", out_dir / (stem + ".json"))
    logger.info("产物 %s", out_dir / (stem + ".xlsx"))
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="整份文档 → json + excel（一页一张大表格）")
    p.add_argument("--input", required=True, type=Path, help="PDF 或图片")
    p.add_argument("--out", type=Path, default=None, help="输出目录，默认 out/<文件名>")
    p.add_argument("--pages", default=None, help="只处理这些页，如 114-116 或 1,3,5")
    p.add_argument("--dpi", type=int, default=None, help="PDF 渲染 dpi，默认取配置")
    p.add_argument("--no-preprocess", action="store_true", help="跳过倾斜/翻转矫正")
    p.add_argument("--backend", choices=BACKENDS, default=None,
                   help="表格识别后端，覆盖 config 的 tablerec.backend")
    p.add_argument("--model-url", default=None, help="覆盖配置里的 paddleocr_vl.base_url")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=Path, default=None,
                   help="日志文件，默认 work/logs/app.log")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)

    from config import load_config
    from logging_setup import setup_logging
    from pdf2image import parse_pages

    cfg = load_config(args.config)
    if args.backend:
        cfg["tablerec"]["backend"] = args.backend
    if args.model_url:
        cfg["tablerec"]["paddleocr_vl"]["base_url"] = args.model_url
    setup_logging(args.log_level, args.log_file)
    logger.info("后端=%s 配置=%s", cfg["tablerec"]["backend"],
                args.config or "config.yaml")
    pages = parse_pages(args.pages)
    out_dir = args.out or (ROOT / cfg["paths"]["out"] / args.input.stem)
    process(args.input, out_dir, cfg, pages=pages, dpi=args.dpi,
            do_preprocess=not args.no_preprocess)
    return 0


if __name__ == "__main__":
    sys.exit(main())
