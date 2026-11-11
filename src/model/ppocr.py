"""PP-OCR ONNX 推理（cv2.dnn，不需要 onnxruntime）。

det: 输出概率图 → 阈值 → 膨胀（近似 unclip）→ 轮廓 → 主成分拟合旋转框
cls: 文本行 3x48x192 → 0°/180° 两类

模型文件放 models/ppocr/，由 config.yaml 指定文件名。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
_CLS_SIZE = (160, 80)          # (w, h)：模型真实输入是 3x80x160


@dataclass
class Box:
    """一个文本行框，points 为 4 个原图像素坐标点。"""

    points: list[tuple[float, float]]
    score: float = 0.0

    @property
    def angle(self) -> float:
        """长边方向，归一化到 [0,180)。0 和 180 是同一个方向。"""
        edges = [_dist(self.points[i], self.points[(i + 1) % 4]) for i in range(4)]
        i = edges.index(max(edges))
        a, b = self.points[i], self.points[(i + 1) % 4]
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0

    @property
    def length(self) -> float:
        edges = [_dist(self.points[i], self.points[(i + 1) % 4]) for i in range(4)]
        return max(edges)

    def to_dict(self) -> dict:
        return {"points": [[round(x, 1), round(y, 1)] for x, y in self.points],
                "score": round(self.score, 4)}


def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _order_quad(points) -> list[tuple[float, float]]:
    """排成 左上、右上、右下、左下。"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums, diffs = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    tl, br = pts[int(np.argmin(sums))], pts[int(np.argmax(sums))]
    tr, bl = pts[int(np.argmin(diffs))], pts[int(np.argmax(diffs))]
    return [tuple(map(float, p)) for p in (tl, tr, br, bl)]


def crop_quad(bgr: np.ndarray, quad) -> np.ndarray:
    """按四点透视裁剪成正立文本行。"""
    tl, tr, br, bl = _order_quad(quad)
    width = int(round(max(_dist(tl, tr), _dist(bl, br))))
    height = int(round(max(_dist(tl, bl), _dist(tr, br))))
    width, height = max(width, 1), max(height, 1)
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (width, height))


def _quad_from_contour(contour: np.ndarray) -> tuple[np.ndarray, float]:
    """轮廓主成分方向拟合旋转矩形，返回 (四角点, 短边)。

    不用 cv2.minAreaRect：它在大致水平的小角度上会量化到 0/90，估角会失效。
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    e0, e1 = eigenvectors[0], eigenvectors[1]
    projected = (pts - mean) @ np.stack([e0, e1], axis=1)
    hw = (projected[:, 0].max() - projected[:, 0].min()) / 2.0
    hh = (projected[:, 1].max() - projected[:, 1].min()) / 2.0
    c = mean.ravel()
    corners = np.array([c - e0 * hw - e1 * hh, c + e0 * hw - e1 * hh,
                        c + e0 * hw + e1 * hh, c - e0 * hw + e1 * hh], dtype=np.float32)
    return corners, float(min(hw, hh) * 2)


def _box_score(prob: np.ndarray, quad: np.ndarray, shrink: float = 0.75) -> float:
    """框内概率均值；框先向内收，避免边界背景拉低分数。"""
    center = quad.mean(axis=0)
    mask = np.zeros(prob.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [(center + (quad - center) * shrink).astype(np.int32)], 1)
    inside = prob[mask == 1]
    return float(inside.mean()) if inside.size else 0.0


class Ppocr:
    def __init__(self, cfg) -> None:
        model_dir = cfg.model_dir if hasattr(cfg, "model_dir") else cfg["model_dir"]
        self.model_dir = Path(str(model_dir))
        self.limit_side = int(cfg.limit_side)
        self.thresh = float(cfg.thresh)
        self.box_thresh = float(cfg.box_thresh)
        self.unclip = max(1, int(round(float(cfg.unclip_ratio))))
        self.min_size = int(cfg.min_size)
        det_path = self.model_dir / cfg.det_model
        cls_path = self.model_dir / cfg.cls_model
        for path in (det_path, cls_path):
            if not path.exists():
                raise FileNotFoundError(f"模型不存在：{path}")
        self.det = cv2.dnn.readNetFromONNX(str(det_path))
        self.cls = cv2.dnn.readNetFromONNX(str(cls_path))

        # 识别模型可选：缺了就只能做检测，不影响矫正链路
        self.rec = None
        self.charset: list[str] = []
        rec_dir = Path(str(cfg.rec_dir)) if ("rec_dir" in cfg) else None
        if rec_dir and (rec_dir / str(cfg["rec_model"])).exists():
            from model.onnx_session import create

            self.rec = create(rec_dir / cfg["rec_model"],
                              providers=("CPUExecutionProvider",))
            dict_path = rec_dir / str(cfg["rec_dict"])
            self.charset = dict_path.read_text(encoding="utf-8").splitlines()

    def detect(self, bgr: np.ndarray) -> list[Box]:
        h, w = bgr.shape[:2]
        scale = min(1.0, self.limit_side / float(max(h, w)))
        net_w = max(32, int(round(w * scale / 32) * 32))
        net_h = max(32, int(round(h * scale / 32) * 32))

        resized = cv2.resize(bgr, (net_w, net_h))
        blob = (resized[:, :, ::-1].astype(np.float32) - _MEAN) / _STD
        self.det.setInput(blob.transpose(2, 0, 1)[None].astype(np.float32))
        prob = self.det.forward()[0, 0]

        seg = cv2.dilate((prob > self.thresh).astype(np.uint8),
                         np.ones((3, 3), np.uint8), iterations=self.unclip)
        contours, _ = cv2.findContours(seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        rx, ry = w / float(prob.shape[1]), h / float(prob.shape[0])
        boxes: list[Box] = []
        for contour in contours:
            if len(contour) < 4:
                continue
            quad, short = _quad_from_contour(contour)
            if short < self.min_size:
                continue
            score = _box_score(prob, quad)
            if score < self.box_thresh:
                continue
            boxes.append(Box(_order_quad([(x * rx, y * ry) for x, y in quad]), score))
        boxes.sort(key=lambda b: -b.length)
        return boxes

    def flip_score(self, bgr: np.ndarray, box: Box) -> float:
        """该文本行"倒置 180°"的概率。"""
        line = crop_quad(bgr, box.points)
        if line.size == 0:
            return 0.0
        blob = (cv2.resize(line, _CLS_SIZE)[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
        self.cls.setInput(blob.transpose(2, 0, 1)[None].astype(np.float32))
        logits = self.cls.forward().ravel()
        exp = np.exp(logits - logits.max())
        return float((exp / exp.sum())[1])

    # ------------------------------------------------------------------ 识别
    def recognize(self, lines: list[np.ndarray], max_ratio: float = 20.0) -> list[str]:
        """单行文本图 → 字符串。CTC 解码，批内按同一宽度补齐。"""
        if self.rec is None:
            return ["" for _ in lines]
        if not lines:
            return []

        heights, widths = [], []
        for line in lines:
            h, w = line.shape[:2]
            nw = max(1, int(round(w * 48.0 / max(h, 1))))
            heights.append(48)
            widths.append(min(nw, int(48 * max_ratio)))
        batch_w = max(widths)
        batch_w = max(16, int(np.ceil(batch_w / 8) * 8))

        blob = np.zeros((len(lines), 3, 48, batch_w), dtype=np.float32)
        for i, line in enumerate(lines):
            resized = cv2.resize(line, (widths[i], heights[i]))
            x = resized[:, :, ::-1].astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            blob[i, :, :, :widths[i]] = x.transpose(2, 0, 1)

        name = self.rec.get_inputs()[0].name
        out = self.rec.run(None, {name: blob})[0]           # [N,T,C]
        return [self._ctc_decode(seq) for seq in out]

    def _ctc_decode(self, seq: np.ndarray) -> str:
        """贪心 CTC：取 argmax，去掉 blank(0) 与连续重复。"""
        idx = seq.argmax(axis=1)
        chars, prev = [], -1
        for k in idx:
            k = int(k)
            if k != 0 and k != prev and k - 1 < len(self.charset):
                chars.append(self.charset[k - 1])
            prev = k
        return "".join(chars)
