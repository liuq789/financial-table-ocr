"""统一的 ONNX 会话创建：关掉内存池，让峰值内存可回收。

为什么单独一个模块
------------------
onnxruntime 默认 `enable_cpu_mem_arena=True` + `kNextPowerOfTwo` 扩展策略：为了减少
malloc 次数，它把每次分配按 2 的幂预占，而且**用完不还给系统**。实测一张 15×7 的
财报页，RSS 从 1.5GB 直接涨到 4.1GB——一个 15GB 的 WSL 撑不住几个这样的峰值，
更何况并发。

关掉 arena 后，张量用完即还（代价是个位数百分比的推理耗时），峰值内存可预测。
所有模型入口都必须走这里，否则一处漏配就白关。

单独跑
------
    PYTHONPATH=src python -m model.onnx_session models/layout/pp_doc_layoutv3.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 本项目所有模型都只用 CPU 或 CUDA；顺序即优先级
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")

# 线程数固定 4：实测 28 核默认线程与 4 线程的推理耗时相同（1.94s vs 2.00s），
# 但每多一个线程就多一份工作内存；常驻服务不需要为了跑分把 28 核全占满。
DEFAULT_THREADS = 4


def create(model_path: Path | str, providers: tuple[str, ...] = PROVIDERS,
           threads: int = DEFAULT_THREADS):
    """建一个"内存可回收"的推理会话。"""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False       # 不用池化分配：峰值内存立刻还回去
    opts.enable_mem_pattern = False         # 不预分配张量图案，按实际需要申请
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), opts, providers=list(providers))


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="建会话并打印 IO，用于单独验证模型可用")
    ap.add_argument("model", type=Path)
    ap.add_argument("--rss", action="store_true", help="顺带打印建会话前后的内存")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from resource_guard import read_rss_mb

    before = read_rss_mb()
    sess = create(args.model)
    print(f"ok  会话已建立（{args.model.name}，providers={sess.get_providers()}）")
    if args.rss:
        print(f"    内存 {before}MB → {read_rss_mb()}MB")
    print("    输入：" + ", ".join(f"{i.name}{i.shape}" for i in sess.get_inputs()))
    print("    输出：" + ", ".join(f"{o.name}{o.shape}" for o in sess.get_outputs()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
