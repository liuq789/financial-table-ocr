"""主干②：图片矫正。

第一步 倾斜矫正（0~180°）
    PP-OCR 检测出带方向的文本框 → 取最长的若干条 → 长边方向归一到 [0,180) →
    圆周中位数 → 旋转。角度不折叠到 ±45，所以 90° 这类大角度也能一次转回来。

第二步 180° 翻转
    检测框裁出文本行送 cls，投票判是否倒置。旋转角在 180° 上不可分
    （θ 和 θ+180 同一方向），所以"正着还是倒着"只能由 cls 决定。

检测只做一次：盒子在两步之间传递，只有真的动了图才重建坐标。
用法：
    PYTHONPATH=src python -m preprocess --pages work/pages/000651_2023_annual
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import ensure_dir, load_config, resolve
from model.ppocr import Box, Ppocr


# ------------------------------------------------------------------ 读写
def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图片：{path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imencode(".png", image)[1].tofile(str(path))


def rotate_image(image: np.ndarray, angle: float, fill: int = 255) -> np.ndarray:
    """旋转并扩画布，不裁内容。"""
    h, w = image.shape[:2]
    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(round(h * sin + w * cos)), int(round(w * sin + h * cos))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=tuple([fill] * 3))


# ------------------------------------------------------------- 角度估计
def circular_median(angles: list[float], period: float = 180.0) -> float:
    """周期量的中位数：先按最近邻展开到同一分支，再取中位数。"""
    values = np.asarray(angles, dtype=np.float64) % period
    reference = float(values[0])
    unwrapped = (values - reference + period / 2) % period - period / 2 + reference
    return float(np.median(unwrapped) % period)


def _longest(boxes: list[Box], cfg) -> list[Box]:
    keep = [b for b in boxes if b.length >= float(cfg.min_length)]
    keep.sort(key=lambda b: -b.length)
    return keep[:int(cfg.top_k)]


def _fold(angle_180: float) -> float:
    """[0,180) 的走向角 → 最小旋转量，落在 (-90, 90]。"""
    return angle_180 - 180.0 if angle_180 > 90.0 else angle_180


def _refine_box(gray: np.ndarray, box: Box, cfg) -> float:
    """单框内搜让行投影最尖锐的角度（±refine_span，步长 refine_step）。"""
    xs = [p[0] for p in box.points]
    ys = [p[1] for p in box.points]
    x0, y0 = max(0, int(min(xs)) - 8), max(0, int(min(ys)) - 8)
    x1, y1 = min(gray.shape[1], int(max(xs)) + 8), min(gray.shape[0], int(max(ys)) + 8)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return _fold(box.angle)

    crop = gray[y0:y1, x0:x1]
    _, binary = cv2.threshold(cv2.GaussianBlur(crop, (3, 3), 0), 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape[:2]
    base = _fold(box.angle)
    best, best_score = base, -1.0
    for angle in np.arange(base - float(cfg.refine_span),
                           base + float(cfg.refine_span) + 1e-9, float(cfg.refine_step)):
        matrix = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), float(angle), 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(rotated.sum(axis=1, dtype=np.float64).var())
        if score > best_score:
            best, best_score = float(angle), score
    return best


def residual_angle(image: np.ndarray, boxes: list[Box], cfg) -> tuple[float, int]:
    """粗转之后剩下的那个小角度（已折叠到 ±45，带框内精修）。"""
    longest = _longest(boxes, cfg)
    if not longest:
        return 0.0, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = np.asarray([_refine_box(gray, b, cfg) for b in longest])
    values = values[np.abs(values) <= 45.0]
    return (float(np.median(values)), int(values.size)) if values.size else (0.0, 0)


# ----------------------------------------------------------------- 第一步
def correct_skew(image: np.ndarray, det: Ppocr, cfg,
                 boxes: list[Box] | None = None) -> tuple[np.ndarray, dict, list[Box]]:
    if boxes is None:
        boxes = det.detect(image)

    longest = _longest(boxes, cfg)
    if not longest:
        return image, {"coarse": 0.0, "refine": 0.0, "total": 0.0,
                       "n_boxes": 0, "applied": False}, boxes

    rotation = _fold(circular_median([b.angle for b in longest]))
    coarse, refine, applied = rotation, 0.0, False

    if abs(rotation) >= float(cfg.min_angle):
        image = rotate_image(image, rotation)
        applied = True
        boxes = det.detect(image)                    # 图变了，坐标必须重建
        refine, n_refine = residual_angle(image, boxes, cfg)
        if abs(refine) >= max(float(cfg.min_angle), float(cfg.residual_min)):
            image = rotate_image(image, refine)
            boxes = det.detect(image)                # 又变了，再建一次
        else:
            refine = 0.0

    total = coarse + refine if applied else 0.0
    return image, {"coarse": round(coarse, 2), "refine": round(refine, 2),
                   "total": round(total, 2), "n_boxes": len(longest),
                   "applied": applied}, boxes


# ----------------------------------------------------------------- 第二步
def boxes_rot180(boxes: list[Box], width: int, height: int) -> list[Box]:
    """检测框随 180° 翻转同步变换，不重新检测。"""
    return [Box([(width - 1.0 - x, height - 1.0 - y) for x, y in b.points], b.score)
            for b in boxes]


def correct_flip(image: np.ndarray, det: Ppocr, cfg,
                 boxes: list[Box] | None = None) -> tuple[np.ndarray, dict, list[Box]]:
    if not boxes:
        return image, {"flipped": False, "confidence": 0.0, "n_boxes": 0}, boxes

    longest = sorted(boxes, key=lambda b: -b.length)[:int(cfg.max_boxes)]
    score = float(np.median([det.flip_score(image, b) for b in longest]))
    info = {"flipped": False, "confidence": round(score, 3), "n_boxes": len(longest)}
    if score < float(cfg.threshold):
        return image, info, boxes

    height, width = image.shape[:2]
    flipped = cv2.rotate(image, cv2.ROTATE_180)
    moved = boxes_rot180(boxes, width, height)

    if bool(cfg.verify):                             # 复检：用变换后的框再打分，不重检
        check = sorted(moved, key=lambda b: -b.length)[:int(cfg.max_boxes)]
        after = float(np.median([det.flip_score(flipped, b) for b in check]))
        if after >= score:
            return image, info, boxes

    info["flipped"] = True
    return flipped, info, moved


# ------------------------------------------------------------------ 批处理
def preprocess_page(path: Path, out_dir: Path, det: Ppocr, cfg) -> dict:
    image = imread(path)
    boxes = det.detect(image)
    skew = flip = None

    if not bool(cfg.skip_skew):
        image, skew, boxes = correct_skew(image, det, cfg.angle, boxes)
    if not bool(cfg.skip_flip):
        image, flip, boxes = correct_flip(image, det, cfg.flip, boxes)

    changed = bool((skew and skew["applied"]) or (flip and flip["flipped"]))
    out_path = out_dir / path.name
    if changed:
        imwrite(out_path, image)
    else:
        ensure_dir(out_dir)
        out_path.write_bytes(path.read_bytes())

    height, width = image.shape[:2]
    ensure_dir(out_dir / "_boxes")
    (out_dir / "_boxes" / f"{path.stem}.json").write_text(json.dumps(
        {"page": path.stem, "width": width, "height": height,
         "boxes": [b.to_dict() for b in boxes]}, ensure_ascii=False), encoding="utf-8")

    return {"page": path.stem, "width": width, "height": height,
            "skew": skew, "flip": flip, "n_boxes": len(boxes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="图片矫正：倾斜 + 180 翻转")
    ap.add_argument("--pages", required=True, help="页面图目录，如 work/pages/<doc>")
    ap.add_argument("--out", default=None, help="输出目录，默认 work/preprocessed/<doc>")
    ap.add_argument("--skip-skew", action="store_true")
    ap.add_argument("--skip-flip", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.skip_skew:
        cfg["preprocess"]["skip_skew"] = True
    if args.skip_flip:
        cfg["preprocess"]["skip_flip"] = True

    src_dir = Path(args.pages)
    files = sorted(src_dir.glob("p*.png"))
    if not files:
        print(f"[error] 目录里没有页面图：{src_dir}", file=sys.stderr)
        return 1
    out_dir = Path(args.out) if args.out else resolve(cfg.paths.work) / "preprocessed" / src_dir.name

    det = Ppocr(cfg.textdet)
    started, report, failed = time.time(), [], 0
    for path in files:
        try:
            item = preprocess_page(path, out_dir, det, cfg.preprocess)
        except Exception as exc:            # noqa: BLE001 单页失败继续
            print(f"[error] {path.name}：{exc}", file=sys.stderr)
            failed += 1
            report.append({"page": path.stem, "error": str(exc)})
            continue
        report.append(item)
        skew = (item["skew"] or {}).get("total", 0.0)
        flipped = (item["flip"] or {}).get("flipped", False)
        print(f"{path.stem} → 旋转 {skew:+.2f}°，翻转 {'是' if flipped else '否'}")

    (out_dir / "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {len(files) - failed}/{len(files)} 页，{time.time() - started:.1f}s → {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
