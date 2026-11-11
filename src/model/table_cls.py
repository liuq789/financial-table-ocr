"""表格类型分类：有线 / 无线（table_cls.onnx）。

实测映射：输出 [2] 的 index 0 = 有线，1 = 无线。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 224


class TableKind:
    def __init__(self, model_path: Path) -> None:
        from model.onnx_session import create

        self.sess = create(model_path)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """返回 ("wired"|"wireless", 置信度)。"""
        x = cv2.resize(image, (SIZE, SIZE))
        x = (x[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        out = self.sess.run(None, {"x": x.transpose(2, 0, 1)[None].astype(np.float32)})[0].ravel()
        idx = int(out.argmax())
        return ("wired" if idx == 0 else "wireless"), float(out[idx])
