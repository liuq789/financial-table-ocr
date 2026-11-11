"""表格结构识别。

两套实现，按表格类型分开：
    有线  WiredUnet        unet.onnx 分割出表格线 → 由线求行/列 → 单元格
    无线  model/lore.py    阿里读光 LORE（ONNX），出四角坐标 + 逻辑行列

对外统一返回 TableStruct（行数列数、展开矩阵、单元格、HTML）。
换实现不影响调用方。

单独跑：
    PYTHONPATH=src python -m model.table_structure <图片> [wired|wireless]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# PaddleX TableLabelDecode(merge_no_span_structure=True) 的词表
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Cell:
    """一个单元格。bbox 为 [x0, y0, x1, y1]，原图坐标。"""

    r: int
    c: int
    rowspan: int = 1
    colspan: int = 1
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class TableStruct:
    kind: str
    n_rows: int
    n_cols: int
    cells: list[Cell] = field(default_factory=list)
    score: float = 0.0
    html: str = ""

    @property
    def matrix(self) -> list[list[int]]:
        """展开成网格，元素是 cells 的下标；被合并覆盖的位置填 -1。"""
        grid = [[-1] * self.n_cols for _ in range(self.n_rows)]
        for i, cell in enumerate(self.cells):
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    r, c = cell.r + dr, cell.c + dc
                    if r < self.n_rows and c < self.n_cols:
                        grid[r][c] = i
        return grid

    def to_html(self) -> str:
        out = ["<table>"]
        grid = self.matrix
        for r in range(self.n_rows):
            out.append("<tr>")
            c = 0
            while c < self.n_cols:
                i = grid[r][c]
                if i < 0:
                    c += 1
                    continue
                cell = self.cells[i]
                if cell.r != r or cell.c != c:      # 被合并覆盖，跳过
                    c += 1
                    continue
                attr = ""
                if cell.colspan > 1:
                    attr += f' colspan="{cell.colspan}"'
                if cell.rowspan > 1:
                    attr += f' rowspan="{cell.rowspan}"'
                out.append(f"<td{attr}></td>")
                c += cell.colspan
            out.append("</tr>")
        out.append("</table>")
        return "".join(out)


def _load(model_path: Path) -> "object":
    from model.onnx_session import create

    return create(model_path)


def _norm_chw(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """BGR → RGB /255 → 减均值除方差 → CHW。size 是 (w, h)。"""
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def _fit_multiple(image: np.ndarray, max_side: int, div: int) -> tuple[np.ndarray, float, float]:
    """等比缩放到长边 ≤ max_side，并把宽高取整到 div 的倍数（UNet 池化要整除）。"""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nw = max(div, int(round(w * scale / div)) * div)
    nh = max(div, int(round(h * scale / div)) * div)
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR), nw / w, nh / h


def _positions(profile: np.ndarray, ratio: float) -> list[int]:
    """把"哪些行/列是线"的剖面压成线位置（相邻的算一条）。"""
    idx = np.flatnonzero(profile > profile.max() * ratio)
    if idx.size == 0:
        return []
    groups, start = [], idx[0]
    for prev, cur in zip(idx[:-1], idx[1:]):
        if cur - prev > 1:
            groups.append((start + prev) // 2)
            start = cur
    groups.append((start + idx[-1]) // 2)
    return groups


def _grid(pos: list[int], size: int, merge_ratio: float = 0.02,
          edge_ratio: float = 0.03) -> list[int]:
    """整理网格线位置：合并过近的线，靠近边缘的吸附成边界，离得远的补边界。

    两个容差必须分开：
      merge_ratio 管"两条线是否算同一条"，行距密所以要小；
      edge_ratio  管"这条线是不是外边框"，裁剪时多留的边距是几个像素到 2%，
                  放宽到 3% 才能吸住，否则会多出一条只有边框没有内容的空行。
    """
    if not pos:
        return [0, size - 1]
    tol = max(2.0, size * merge_ratio)
    merged = [float(pos[0])]
    for p in pos[1:]:
        if p - merged[-1] > tol:
            merged.append(float(p))
        else:
            merged[-1] = (merged[-1] + p) / 2
    edge = max(4.0, size * edge_ratio)
    if merged[0] <= edge:
        merged[0] = 0.0
    else:
        merged.insert(0, 0.0)         # 边框线没检出来，必须补，否则少一列
    if size - 1 - merged[-1] <= edge:
        merged[-1] = float(size - 1)
    else:
        merged.append(float(size - 1))
    return [int(v) for v in merged]


class WiredUnet:
    """有线表格：unet 输出线掩码 → 由线求行列 → 单元格。"""

    def __init__(self, model_path: Path) -> None:
        self.sess = _load(model_path)

    def predict(self, image: np.ndarray) -> TableStruct:
        h, w = image.shape[:2]
        canvas, sx, sy = _fit_multiple(image, max_side=1024, div=32)
        mask = np.asarray(self.sess.run(None, {"input": _norm_chw(canvas, (canvas.shape[1],
                                                                          canvas.shape[0]))})[0])
        lines = (mask[0, 0] > 0).astype(np.uint8)

        # 横线 / 竖线分开：用长核做开运算，短线（文字残留）会被滤掉
        kw = max(15, canvas.shape[1] // 25)
        kh = max(15, canvas.shape[0] // 25)
        horiz = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((1, kw), np.uint8))
        vert = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((kh, 1), np.uint8))

        ch, cw = canvas.shape[:2]
        # 阈值 0.45 在密集的财报表上会漏掉浅色行线（p115 只检出 35 条，真实 ~50），
        # 降到 0.30 多召回几条；有长核开运算兜着，不会把文字当成线。
        ys = _positions(horiz.sum(axis=1).astype(np.float32), 0.30)
        xs = _positions(vert.sum(axis=0).astype(np.float32), 0.30)
        # 表格外边框就是裁剪框边缘，剖面里常常检不出来，必须补上，
        # 否则会少两个边界 → 行列数直接少 2
        ys = _grid([p for p in ys if p < ch], ch, merge_ratio=0.006)   # 行距小，别合并
        xs = _grid([p for p in xs if p < cw], cw, merge_ratio=0.02)
        if len(ys) < 2 or len(xs) < 2:                 # 线没检出来，退回整块
            ys = [0, ch - 1]
            xs = [0, cw - 1]

        n_rows, n_cols = len(ys) - 1, len(xs) - 1
        cells = [Cell(r=r, c=c,
                      bbox=(int(xs[c] / sx), int(ys[r] / sy),
                            int(xs[c + 1] / sx), int(ys[r + 1] / sy)))
                 for r in range(n_rows) for c in range(n_cols)]

        # 合并单元格：某条内部线不存在 → 两边其实是一格
        cells = self._merge(cells, lines, ys, xs)
        # 合并只改了 rowspan/colspan，bbox 必须按合并后的范围重算，
        # 否则整格 OCR 只裁到原来那半格（表头"项目"会被切成一条缝）
        for cell in cells:
            cell.bbox = (int(xs[cell.c] / sx), int(ys[cell.r] / sy),
                         int(xs[cell.c + cell.colspan] / sx),
                         int(ys[cell.r + cell.rowspan] / sy))
        st = TableStruct(kind="wired", n_rows=n_rows, n_cols=n_cols, cells=cells)
        st.html = st.to_html()
        return st

    @staticmethod
    def _merge(cells: list[Cell], lines: np.ndarray, ys: list[int], xs: list[int]) -> list[Cell]:
        """沿网格逐行合并：相邻两格之间的线上没有墨迹就合并。"""
        grid = {(c.r, c.c): c for c in cells}
        thick = 3

        def has_sep(horizontal: bool, pos_idx: int, lo: int, hi: int) -> bool:
            """在 pos_idx 处、[lo,hi) 区间内是否存在分隔线。"""
            if horizontal:
                y = ys[pos_idx]
                band = lines[max(0, y - thick): y + thick + 1, lo:hi]
                return band.size > 0 and float((band > 0).mean()) > 0.25
            x = xs[pos_idx]
            band = lines[lo:hi, max(0, x - thick): x + thick + 1]
            return band.size > 0 and float((band > 0).mean()) > 0.25

        merged: list[Cell] = []
        taken: set[tuple[int, int]] = set()
        for r in range(len(ys) - 1):
            for c in range(len(xs) - 1):
                if (r, c) in taken:
                    continue
                cell = grid[(r, c)]
                # 向右扩
                while (c + cell.colspan < len(xs) - 1
                       and not has_sep(False, c + cell.colspan, ys[r], ys[r + 1])):
                    cell.colspan += 1
                # 向下扩（只在 colspan 覆盖范围内都没有横线时才算合并）
                while cell.r + cell.rowspan < len(ys) - 1:
                    y_bottom = cell.r + cell.rowspan
                    if has_sep(True, y_bottom, xs[c], xs[c + cell.colspan]):
                        break
                    cell.rowspan += 1
                for dr in range(cell.rowspan):
                    for dc in range(cell.colspan):
                        taken.add((r + dr, c + dc))
                merged.append(cell)
        return merged


def build(kind: str, cfg) -> "WiredUnet | object":
    """有线走 unet；无线走读光（独立模块，ONNX 推理）。"""
    ts = cfg["tablestruct"] if isinstance(cfg, dict) else cfg.tablestruct
    if kind == "wired":
        return WiredUnet(ROOT / ts["model_dir"] / ts["wired_model"])
    from model.lore import LoreOnnx

    return LoreOnnx(ROOT / ts["lore_dir"])


def auto_crop(image: np.ndarray, cfg) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """用 PP-OCR 的检测框并集估出表格区域（没有版面模型时的兜底）。"""
    from model.ppocr import Ppocr

    boxes = Ppocr(cfg.textdet).detect(image)
    if not boxes:
        return image, (0, 0, image.shape[1], image.shape[0])
    pts = np.concatenate([np.array(b.points) for b in boxes])
    x0, y0 = pts.min(0).astype(int)
    x1, y1 = pts.max(0).astype(int)
    m = int(image.shape[1] * 0.01)
    x0, y0 = max(0, x0 - m), max(0, y0 - m)
    x1, y1 = min(image.shape[1], x1 + m), min(image.shape[0], y1 + m)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    import argparse

    from config import load_config

    p = argparse.ArgumentParser(description="表格结构识别")
    p.add_argument("image", type=Path)
    p.add_argument("kind", nargs="?", default="wired", choices=["wired", "wireless"])
    p.add_argument("--crop", help="手工指定表格区域 x0,y0,x1,y1")
    p.add_argument("--auto-crop", action="store_true", help="用 PP-OCR 检测框自动裁")
    p.add_argument("--out", type=Path, default=None, help="可视化输出路径")
    args = p.parse_args(argv)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"读不了图片：{args.image}")
        return 1

    cfg = load_config()
    offset = (0, 0)
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
        image = image[y0:y1, x0:x1]
        offset = (x0, y0)
    elif args.auto_crop:
        image, box = auto_crop(image, cfg)
        offset = (box[0], box[1])
        print(f"自动裁图 {box} → {image.shape[1]}x{image.shape[0]}")

    struct = build(args.kind, cfg).predict(image)
    print(f"{args.kind}: {struct.n_rows} 行 × {struct.n_cols} 列，{len(struct.cells)} 个单元格"
          f"，score {struct.score:.3f}")
    print(struct.html[:300])

    vis = image.copy()
    for cell in struct.cells:
        x0, y0, x1, y1 = cell.bbox
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
    out = args.out or ROOT / "work" / f"struct_{args.kind}.png"
    cv2.imwrite(str(out), vis)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
