"""冒烟测试（不依赖 pytest）。

用法：python tests/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config, resolve                        # noqa: E402
from model.ppocr import Ppocr                                  # noqa: E402
from pdf2image import render_pdf                               # noqa: E402
from preprocess import correct_flip, correct_skew, imread, rotate_image  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def main() -> int:
    cfg = load_config()
    work = resolve(cfg.paths.work)
    pdf = resolve(cfg.paths.data) / "pdfs" / "000651_2023_annual.pdf"

    pages = render_pdf(pdf, work / "_smoke" / "pages", cfg, pages=[114])
    check("PDF → 图片", len(pages) == 1, f"{pages[0]['width']}x{pages[0]['height']}")

    det = Ppocr(cfg.textdet)
    page = imread(Path(pages[0]["path"]))

    boxes = det.detect(page)
    check("文字检测", len(boxes) > 20, f"{len(boxes)} 框")
    check("框带方向", all(0 <= b.angle < 180 for b in boxes))

    _, skew, kept = correct_skew(page, det, cfg.preprocess.angle)
    check("直页不转", abs(skew["total"]) < 0.5, f"{skew['total']:+.2f}°")
    check("倾斜矫正返回框", len(kept) > 20, f"{len(kept)} 框")

    for applied in (1.5, -4.0, 7.0):
        _, info, _ = correct_skew(rotate_image(page, applied), det, cfg.preprocess.angle)
        check(f"倾斜 {applied:+.1f}° 被修正", abs(info["total"] + applied) < 0.4,
              f"{info['total']:+.2f}°")

    for name, flag in (("顺时针90", cv2.ROTATE_90_CLOCKWISE),
                       ("逆时针90", cv2.ROTATE_90_COUNTERCLOCKWISE)):
        fixed, info, _ = correct_skew(cv2.rotate(page, flag), det, cfg.preprocess.angle)
        check(f"{name} 转回竖版", fixed.shape[0] > fixed.shape[1], f"{info['total']:+.1f}°")

    _, info, same = correct_flip(page, det, cfg.preprocess.flip, boxes)
    check("正常页不翻", not info["flipped"], f"置信 {info['confidence']:.2f}")
    check("不翻时复用传入框", len(same) == len(boxes))

    upside = cv2.rotate(page, cv2.ROTATE_180)
    fixed, info, moved = correct_flip(upside, det, cfg.preprocess.flip, det.detect(upside))
    check("180 页被翻正", info["flipped"], f"置信 {info['confidence']:.2f}")
    check("框随图变换", min(det.flip_score(fixed, b) for b in moved[:5]) < 0.4)

    print(f"\n{'全部通过' if not fails else f'{len(fails)} 项失败：{fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
