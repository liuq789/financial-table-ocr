"""阿里读光 LORE —— 无线表格结构识别，ONNX 推理。

独立实现，不依赖 modelscope / torch / GPU：只读 models/lore/ 下的 det.onnx 和 proc.onnx，
中间那套 topk / NMS / 坐标反变换 / 特征聚合用 numpy 复刻。

网络分工（导出见 src/tools/export_lore_onnx.py）：
    det.onnx   img[1,3,768,768] → hm[2] st[8] wh[8] ax[256] cr[256] reg[2]（均为 192×192）
    proc.onnx  feat[1,N,256] + dets[1,N,8] → logic_axis[1,N,4] + stacked_axis[1,N,4]

输出：每个单元格四角坐标（原图像素）+ 逻辑坐标 (row_start, row_end, col_start, col_end)。

单独跑：
    PYTHONPATH=src python -m model.lore <图片>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INP = 768                 # 网络输入边长
OUT = INP // 4            # 输出热图边长 192
MEAN = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
K_CORNER = 5000           # 角点候选数
K_CENTER = 3000           # 单元格候选数
SCORE_THRESH = 0.15       # 单元格得分下限（与官方 filter 一致）
FEAT_DIM = 256


@dataclass
class LoreCell:
    """一个单元格：四角坐标（原图）+ 逻辑行列（右下开区间）。"""

    poly: np.ndarray              # (4,2) float
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    def to_dict(self) -> dict:
        return {"poly": self.poly.astype(int).tolist(),
                "row": [self.row_start, self.row_end],
                "col": [self.col_start, self.col_end]}


# --------------------------------------------------------------------- 前处理

def preprocess(image: np.ndarray) -> tuple[np.ndarray, float]:
    """等比缩放到长边 768，左上角对齐，归一化。

    必须用 warpAffine + 同一个矩阵（等价于 768/s 的纯缩放），不能用
    resize+padding：两者半像素采样约定不同，会让检测器输出整体偏 1px，
    进而让 topk 选到不同像素、行号错位。
    """
    h, w = image.shape[:2]
    k = INP / float(max(h, w))
    m = np.array([[k, 0, 0], [0, k, 0]], dtype=np.float32)
    warped = cv2.warpAffine(image, m, (INP, INP), flags=cv2.INTER_LINEAR)
    x = ((warped / 255.0 - MEAN) / STD).astype(np.float32)
    return x.transpose(2, 0, 1)[None], k


# ------------------------------------------------------------------ 后处理基础

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _nms(heat: np.ndarray) -> np.ndarray:
    """3×3 最大池化做 NMS：只保留局部极大。heat 形状 [H,W]。"""
    kernel = np.ones((3, 3), dtype=np.uint8)
    hmax = cv2.dilate(heat.astype(np.float32), kernel,
                      borderType=cv2.BORDER_CONSTANT, borderValue=-1e30)
    return heat * (hmax == heat)


def _topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """展平后取前 k 大，返回 (值, 下标)。同分时取下标小的，与 torch.topk 行为一致。"""
    flat = scores.reshape(-1)
    k = min(k, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    order = np.lexsort((idx, -flat[idx]))
    idx = idx[order]
    return flat[idx], idx


def _gather_map(feat: np.ndarray, inds: np.ndarray) -> np.ndarray:
    """feat [C,H,W] 按展平下标取列，返回 [K,C]。"""
    c = feat.shape[0]
    return feat.reshape(c, -1)[:, inds].T


def _decode_cells(heat: np.ndarray, wh: np.ndarray, ax: np.ndarray, cr: np.ndarray,
                  reg: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单元格解码：返回 (dets[K,8] 192 空间坐标, logi_feat[K,256], cc_feat[K,256])。"""
    h, w = heat.shape
    heat = _nms(heat)
    scores, inds = _topk(heat, K_CENTER)

    reg_k = _gather_map(reg, inds)
    xs = (inds % w).astype(np.float32) + reg_k[:, 0]
    ys = (inds // w).astype(np.float32) + reg_k[:, 1]

    wh_k = _gather_map(wh, inds)                   # [K,8]
    dets = np.stack([xs - wh_k[:, 0], ys - wh_k[:, 1],
                     xs - wh_k[:, 2], ys - wh_k[:, 3],
                     xs - wh_k[:, 4], ys - wh_k[:, 5],
                     xs - wh_k[:, 6], ys - wh_k[:, 7]], axis=1)

    # 四个角在展平特征图里的下标，用来聚合角点特征 cr
    cc = np.stack([xs - wh_k[:, 0] + w * np.round(ys - wh_k[:, 1]),
                   xs - wh_k[:, 2] + w * np.round(ys - wh_k[:, 3]),
                   xs - wh_k[:, 4] + w * np.round(ys - wh_k[:, 5]),
                   xs - wh_k[:, 6] + w * np.round(ys - wh_k[:, 7])], axis=1)
    cc = np.round(cc).astype(np.int64)
    cc = np.where((cc >= 0) & (cc < h * w), cc, 0)      # 官方实现同样把越界归 0

    cr_flat = cr.reshape(FEAT_DIM, -1)                  # [256, H*W]
    cc_feat = cr_flat[:, cc].sum(axis=2).T              # [256,K,4] → 四角求和 → [K,256]
    logi_feat = _gather_map(ax, inds)                   # [K,256]
    return dets, logi_feat, cc_feat, scores


# --------------------------------------------------------------------- 推理

class LoreOnnx:
    """读光：det.onnx + proc.onnx，纯 onnxruntime 推理。"""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):        # CUDA 的 cuDNN 在 site-packages 里
            ort.preload_dlls()

        model_dir = Path(model_dir)
        det, proc = model_dir / "det.onnx", model_dir / "proc.onnx"
        for p in (det, proc):
            if not p.exists():
                raise FileNotFoundError(f"缺读光模型 {p}（先跑 src/tools/export_lore_onnx.py）")

        from model.onnx_session import create

        providers = ("CPUExecutionProvider",)
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ("CUDAExecutionProvider",) + providers
        self.det = create(det, providers=providers)
        self.proc = create(proc, providers=providers)

    def predict(self, image: np.ndarray) -> list[LoreCell]:
        h, w = image.shape[:2]
        x, scale = preprocess(image)

        outs = self.det.run(None, {"img": x})
        head = dict(zip([o.name for o in self.det.get_outputs()], outs))
        hm = _sigmoid(np.asarray(head["hm"])[0])        # [2,H,W]

        dets, logi_feat, cc_feat, scores = _decode_cells(
            hm[0], np.asarray(head["wh"])[0], np.asarray(head["ax"])[0],
            np.asarray(head["cr"])[0], np.asarray(head["reg"])[0])

        keep = scores >= SCORE_THRESH
        if keep.sum() == 0:
            return []
        dets, feat = dets[keep], (logi_feat + cc_feat)[keep]

        # 坐标：热图 192 空间 → 768 画布（×4）→ 原图（÷scale）
        polys = (dets.reshape(-1, 4, 2) * (INP / OUT / scale)).astype(np.float32)

        # 4) 逻辑位置回归。
        #    官方实现把 float 赋给 np.int32 数组，是**截断**而不是四舍五入，
        #    这里必须一致，否则位置嵌入整体偏 1，行号会系统性 +1。
        idx = np.clip(dets.astype(np.int64), 0, FEAT_DIM - 1)
        # 输出顺序是 (base, stacked)，官方取的是 stacked（精修后的行列）
        logi = self.proc.run(None, {"feat": feat[None].astype(np.float32),
                                    "dets": idx[None]})[1][0]
        logi = np.where(logi - np.floor(logi) > 0.5, np.floor(logi) + 1, np.floor(logi))

        cells = []
        for poly, box in zip(polys, logi.astype(int)):
            r0, r1, c0, c1 = (int(v) for v in box[:4])
            cells.append(LoreCell(poly=poly, row_start=r0, row_end=r1,
                                  col_start=c0, col_end=c1))
        return cells


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from config import load_config

    path = Path(argv[0]) if argv else ROOT / "work" / "_wl1.png"
    image = cv2.imread(str(path))
    if image is None:
        print(f"读不了 {path}")
        return 1

    cfg = load_config()
    model = LoreOnnx(ROOT / cfg["tablestruct"]["lore_dir"])
    cells = model.predict(image)
    print(f"{path.name}: {len(cells)} 个单元格")
    if cells:
        rows = max(c.row_end for c in cells) + 1
        cols = max(c.col_end for c in cells) + 1
        print(f"逻辑范围 {rows} 行 × {cols} 列")
        print("前 3 个:", [c.to_dict() for c in cells[:3]])

    vis = image.copy()
    for cell in cells:
        cv2.polylines(vis, [cell.poly.astype(np.int32)], True, (0, 0, 255), 2)
    out = ROOT / "work" / f"lore_onnx_{path.stem}.png"
    cv2.imwrite(str(out), vis)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
