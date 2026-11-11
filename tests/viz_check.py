"""造测试数据 + 画中间检测结果（人工看图用）。

输出 work/viz/：
    {case}_1_detect.png  检测框（整页 + 1:1 放大）
    {case}_2_skew.png    倾斜矫正前后（红横线是水平基准）
    {case}_3_flip.png    翻转判定（每行倒置概率 + 最终输出）
    overview.png         全部用例总览

用法：python tests/viz_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ensure_dir, load_config, resolve      # noqa: E402
from model.ppocr import Box, Ppocr                       # noqa: E402
from preprocess import correct_flip, correct_skew, imread, rotate_image  # noqa: E402

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
GREEN, AMBER, RED, BLUE, BG = (16, 145, 80), (205, 130, 15), (205, 45, 45), (25, 85, 205), (28, 32, 38)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=0)


def to_rgb(bgr: np.ndarray, max_width: int | None = None):
    scale = 1.0
    if max_width and bgr.shape[1] > max_width:
        scale = max_width / bgr.shape[1]
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), scale


def band(img: Image.Image, lines: list[str]) -> Image.Image:
    header = 20 + 30 * len(lines)
    canvas = Image.new("RGB", (img.width, img.height + header), BG)
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(lines):
        draw.text((14, 10 + 30 * i), text, font=font(21), fill=(240, 240, 240))
    canvas.paste(img, (0, header))
    return canvas


def tag(draw: ImageDraw.ImageDraw, xy, text: str, color) -> None:
    f = font(16)
    box = draw.textbbox((0, 0), text, font=f)
    x, y = xy
    draw.rectangle([x - 2, y - 1, x + box[2] + 2, y + box[3] + 4], fill="white")
    draw.text((x, y), text, font=f, fill=color)


def draw_boxes(img, boxes, scale, offset=(0, 0), highlight=None, probs=None, label=False):
    draw = ImageDraw.Draw(img)
    ox, oy = offset
    for i, box in enumerate(boxes):
        pts = [((x - ox) * scale, (y - oy) * scale) for x, y in box.points]
        if probs and i in probs:
            color, thick = (RED if probs[i] >= 0.5 else GREEN), 3
        elif highlight and i in highlight:
            color, thick = BLUE, 3
        else:
            color, thick = (GREEN if min(box.angle, 180 - box.angle) < 1.0 else AMBER), 2
        draw.polygon(pts, outline=color, width=thick)
        y = max(pts[0][1] - 20, 2)
        if label and highlight and i in highlight:
            deg = box.angle - 180 if box.angle > 90 else box.angle
            tag(draw, (pts[0][0], y), f"{deg:+.1f}°", BLUE)
        if probs and i in probs:
            tag(draw, (pts[0][0], y), f"{probs[i]:.2f}", RED if probs[i] >= 0.5 else GREEN)


def guides(panel: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    for ratio in (0.30, 0.50, 0.70):
        y = int(panel.height * ratio)
        draw.line([(0, y), (panel.width, y)], fill=(220, 60, 60), width=2)
    return panel


def hstack(panels, gap=16) -> Image.Image:
    canvas = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1),
                               max(p.height for p in panels)), (238, 238, 238))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def fit(img: Image.Image, height: int) -> Image.Image:
    return img.resize((int(img.width * height / img.height), height))


def main() -> int:
    cfg = load_config()
    out = ensure_dir(resolve(cfg.paths.work) / "viz")
    det = Ppocr(cfg.textdet)
    page = imread(resolve(cfg.paths.work) / "pages" / "000651_2023_annual" / "p114.png")

    cases = {
        "orig": page,
        "rot+3": rotate_image(page, 3.0),
        "rot-6": rotate_image(page, -6.0),
        "rot90cw": cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE),
        "rot180": cv2.rotate(page, cv2.ROTATE_180),
    }
    strips = []

    for name, image in cases.items():
        safe = name.replace("+", "p").replace("-", "m")
        boxes = det.detect(image)

        # 1 检测：整页 + 1:1
        full, s1 = to_rgb(image, max_width=760)
        draw_boxes(full, boxes, s1)
        full = band(full, [f"{name} · 检测 {len(boxes)} 框", "绿=近水平 橙=有角度"])

        h, w = image.shape[:2]
        x0, y0 = int(w * 0.04), int(h * 0.05)
        zoom, s2 = to_rgb(image[y0:y0 + 620, x0:x0 + 940])
        top = set(sorted(range(len(boxes)), key=lambda i: -boxes[i].length)[:8])
        draw_boxes(zoom, boxes, s2, offset=(x0, y0), highlight=top, label=True)
        zoom = band(zoom, ["1:1 放大 · 蓝框=参与估角的 8 条最长文本", "标签为该行方向 -45~45°"])
        hstack([full, zoom]).save(out / f"{safe}_1_detect.png")

        # 2 倾斜
        skewed, skew, boxes2 = correct_skew(image, det, cfg.preprocess.angle, boxes)
        left = band(guides(fit(to_rgb(image, max_width=620)[0], 700)), ["矫正前（红=水平基准）"])
        right = band(guides(fit(to_rgb(skewed, max_width=620)[0], 700)),
                     [f"{name} · 矫正后 {skew['total']:+.2f}°"])
        panel = band(hstack([left, right]),
                     [f"{name} · 倾斜矫正 粗估 {skew['coarse']:+.2f}° "
                      f"残差 {skew['refine']:+.2f}° 合计 {skew['total']:+.2f}°",
                      f"用 {skew['n_boxes']} 条最长文本的圆周中位数估算"])
        panel.save(out / f"{safe}_2_skew.png")

        # 3 翻转（必须在翻转前的图 + 翻转前的框上算，否则坐标对不上）
        flipped, flip, _ = correct_flip(skewed, det, cfg.preprocess.flip, boxes2)
        idx = sorted(range(len(boxes2)), key=lambda i: -boxes2[i].length)[:int(cfg.preprocess.flip.max_boxes)]
        probs = {i: det.flip_score(skewed, boxes2[i]) for i in idx}

        left = to_rgb(skewed, max_width=700)[0]
        draw_boxes(left, boxes2, 700 / skewed.shape[1], probs=probs)
        left = band(left, [f"{name} · 翻转判定 中位 {flip['confidence']:.2f} → "
                           f"{'翻 180°' if flip['flipped'] else '不翻'}",
                           f"阈值 {cfg.preprocess.flip.threshold} 红=倒置 绿=正常"])

        zh, s3 = to_rgb(skewed[y0:y0 + 560, x0:x0 + 900])
        draw_boxes(zh, boxes2, s3, offset=(x0, y0), probs=probs)
        zh = band(zh, ["1:1 放大 · 数字=该行倒置概率"])

        final = band(fit(to_rgb(flipped, max_width=560)[0], 700),
                     [f"{name} · 最终 {'已翻 180°' if flip['flipped'] else '未翻转'}"])
        hstack([left, zh, final]).save(out / f"{safe}_3_flip.png")

        strips.append(hstack([fit(full, 470), fit(right, 470), fit(final, 470)]))
        print(f"{name:8} 框 {len(boxes):3d}  粗估 {skew['coarse']:+7.2f}°  "
              f"残差 {skew['refine']:+5.2f}°  合计 {skew['total']:+7.2f}°  "
              f"翻转 {flip['confidence']:.2f} → {'是' if flip['flipped'] else '否'}")

    width = 1440
    thumbs = [fit(s, int(s.height * width / s.width)) for s in strips]
    overview = Image.new("RGB", (width, sum(t.height for t in thumbs) + 10 * len(thumbs)), (232, 232, 232))
    y = 0
    for thumb in thumbs:
        overview.paste(thumb, (0, y))
        y += thumb.height + 10
    overview.save(out / "overview.png")
    print(f"\n输出：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
