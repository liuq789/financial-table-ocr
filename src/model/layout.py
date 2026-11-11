"""版面分析（PP-DocLayoutV3）：只取表格区域。

模型输出 [N,7] = (类别, 分数, x1, y1, x2, y2, 附加)，坐标已还原到原图像素。
实测三份年报里表格类的 id 是 21（得分 0.75~0.96，覆盖整张表），其余类别
（标题/正文/页码等）一律不要。

单独跑：
    PYTHONPATH=src python -m model.layout <图片>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIZE = 800
TABLE_CLASS = 21                    # PP-DocLayoutV3 里表格的类别 id
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


class Layout:
    def __init__(self, model_path: Path, score_thresh: float = 0.5) -> None:
        from model.onnx_session import create

        self.sess = create(model_path)
        self.score_thresh = score_thresh

    def detect_candidates(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """所有表格类候选框，**含低于阈值的**，按分数从高到低。

        为什么要把低分框也交出去：宽而扁、大量留白的权益变动表 / 现金流量表，
        模型只给 0.37~0.39（p133/p135 实测），按 0.5 卡会整张表丢失。上层可以
        用文字排布复核这些候选（行多、列多、左边缘对齐），通过再采纳。
        """
        h, w = image.shape[:2]
        ratio = min(SIZE / h, SIZE / w)
        nh, nw = max(1, int(round(h * ratio))), max(1, int(round(w * ratio)))
        canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        x = ((canvas[:, :, ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {
            "image": x.astype(np.float32),
            "im_shape": np.array([[SIZE, SIZE]], dtype=np.float32),
            "scale_factor": np.array([[ratio, ratio]], dtype=np.float32),
        })

        boxes = np.asarray(outs[0])
        if boxes.size == 0:
            return []
        keep = boxes[:, 0].astype(int) == TABLE_CLASS
        result = []
        for _, score, x1, y1, x2, y2, _extra in boxes[keep]:
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w, x2)))
            y2 = int(max(0, min(h, y2)))
            if x2 - x1 < 40 or y2 - y1 < 30:      # 太小的当噪声
                continue
            result.append(((x1, y1, x2, y2), float(score)))
        result.sort(key=lambda item: (-item[1], -(item[0][2] - item[0][0])))
        return result

    def detect_tables(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """返回表格框 [(x1,y1,x2,y2)]，按面积从大到小（只留分数达标的）。"""
        rects = [rect for rect, score in self.detect_candidates(image)
                 if score >= self.score_thresh]
        return sorted(rects, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from config import load_config, resolve

    if not argv:
        print(__doc__)
        return 2
    cfg = load_config()
    model_path = resolve(cfg.textdet.model_dir).parent / cfg["layout"]["model"]

    image = cv2.imread(str(argv[0]))
    if image is None:
        print(f"读不了 {argv[0]}")
        return 1
    boxes = Layout(model_path).detect_tables(image)
    print(f"{Path(argv[0]).name}: {len(boxes)} 个表格")
    for b in boxes:
        print("  ", b)

    vis = image.copy()
    for b in boxes:
        cv2.rectangle(vis, b[:2], b[2:], (0, 140, 255), 4)
    out = ROOT / "work" / f"layout_{Path(argv[0]).stem}.png"
    cv2.imwrite(str(out), vis)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
