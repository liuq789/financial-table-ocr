"""检查所有 ONNX 模型的可用性与输入输出形状（不跑推理）。

用法：python tests/check_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config, resolve      # noqa: E402


def describe(path: Path, providers: list[str]) -> None:
    print(f"\n{path.relative_to(resolve(cfg.paths.work).parents[0])}  ({path.stat().st_size / 1024 / 1024:.1f}MB)")
    try:
        session = ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:                    # noqa: BLE001
        print(f"  FAIL: {str(exc)[:200]}")
        return
    print(f"  provider: {session.get_providers()[0]}")
    for item in session.get_inputs():
        print(f"  in   {item.name:<24} {item.shape} {item.type}")
    for item in session.get_outputs():
        print(f"  out  {item.name:<24} {item.shape} {item.type}")


if __name__ == "__main__":
    cfg = load_config()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model_dir = resolve(cfg.textdet.model_dir).parent
    for name in ("layout/pp_doc_layoutv3.onnx", "table/table_cls.onnx",
                 "table/unet.onnx", "lore/det.onnx", "lore/proc.onnx",
                 "ocr/PP-OCRv6_rec_small.onnx",
                 "ppocr/PP-OCRv6_det_small.onnx",
                 "ppocr/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"):
        path = model_dir / name
        if not path.exists():
            print(f"\n{name}  缺失")
            continue
        describe(path, providers)
