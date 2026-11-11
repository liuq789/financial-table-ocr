"""资源预算：在"开始干活"之前就判断内存/显存够不够，不够就快速拒绝。

为什么要有它
------------
"跑挂了再说"在长时间驻留的服务里等于把整个进程交给内核 OOM killer 处理——
十几秒后进程消失，所有在排队的人一起失败。这里的原则是：

1. **只允许一种失败**：单个请求被拒（429/503 + 明确原因），进程本身永远活着。
2. **拒绝发生在动手之前**：先读 /proc/meminfo 与 nvidia-smi，余量不够直接拒。
3. **任务进行中也复查**：每页之间查一次，涨得太猛就中止这一单，而不是等 OOM。

读的都是系统真实数据，不用估算的经验值。

单独跑
------
    PYTHONPATH=src python -m resource_guard            # 打一行当前资源
    PYTHONPATH=src python -m resource_guard --loop 5   # 每 5 秒一行，观察趋势
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MB = 1024 * 1024


class ResourceShortage(RuntimeError):
    """内存或显存不够，拒绝开工。上层应转成 429/503，而不是让它冒成 500。"""


@dataclass(frozen=True)
class Memory:
    total_mb: int
    available_mb: int
    swap_free_mb: int

    def as_dict(self) -> dict:
        return {"total_mb": self.total_mb, "available_mb": self.available_mb,
                "swap_free_mb": self.swap_free_mb}


@dataclass(frozen=True)
class Gpu:
    name: str
    total_mb: int
    free_mb: int

    def as_dict(self) -> dict:
        return {"name": self.name, "total_mb": self.total_mb, "free_mb": self.free_mb}


def read_memory() -> Memory:
    """读 /proc/meminfo。取不到就抛，绝不返回假数据。"""
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as fp:
        for line in fp:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                values[key] = int(parts[0])      # kB
    return Memory(total_mb=values["MemTotal"] // 1024,
                  available_mb=values["MemAvailable"] // 1024,
                  swap_free_mb=values.get("SwapFree", 0) // 1024)


def read_rss_mb() -> int:
    """当前进程常驻内存。VmRSS 在 /proc/self/status 里。"""
    with open("/proc/self/status", encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return 0


def read_gpu() -> Gpu | None:
    """有 nvidia-smi 就返回第一块卡的余量；没有（或没驱动）返回 None。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("读显存失败：%s", exc)
        return None
    if not out:
        return None
    name, total, free = [part.strip() for part in out[0].split(",")]
    return Gpu(name=name, total_mb=int(total), free_mb=int(free))


@dataclass(frozen=True)
class Limits:
    """资源门槛，全部来自 config.yaml 的 guard 段。"""

    min_free_mb: int = 1200       # 开工前系统可用内存下限
    min_gpu_free_mb: int = 1024   # 模型服务在跑时，显存至少留这么多
    max_rss_mb: int = 4096        # 单任务允许的最大常驻内存，超了中止这一单
    max_page_pixels: int = 40_000_000   # 单页像素上限（约 40MP ≈ 120MB BGR）
    max_pages: int = 60

    @classmethod
    def from_config(cls, cfg: dict) -> "Limits":
        raw = (cfg or {}).get("guard") or {}
        return cls(min_free_mb=int(raw.get("min_free_mb", 1200)),
                   min_gpu_free_mb=int(raw.get("min_gpu_free_mb", 1024)),
                   max_rss_mb=int(raw.get("max_rss_mb", 4096)),
                   max_page_pixels=int(raw.get("max_page_pixels", 40_000_000)),
                   max_pages=int(raw.get("max_pages", 60)))


class Budget:
    """一次任务的资源配额：开工前验一次，每页之间再验一次。"""

    def __init__(self, limits: Limits, tag: str = "") -> None:
        self.limits = limits
        self.tag = tag
        self.started_rss_mb = read_rss_mb()

    def precheck(self) -> Memory:
        """开工前：系统可用内存和（可选）显存都要够。"""
        mem = read_memory()
        if mem.available_mb < self.limits.min_free_mb:
            raise ResourceShortage(
                f"系统可用内存只剩 {mem.available_mb}MB（要求 ≥{self.limits.min_free_mb}MB），"
                f"为避免整机 OOM 拒绝这次任务，请稍后重试")
        gpu = read_gpu()
        # 只有模型服务真在跑的时候才要求显存余量；纯 legacy 不用 GPU
        if gpu is not None and gpu.free_mb < self.limits.min_gpu_free_mb:
            logger.warning("显存只剩 %dMB（阈值 %dMB），本次任务将不依赖 GPU 的部分照跑",
                           gpu.free_mb, self.limits.min_gpu_free_mb)
        return mem

    def check(self, stage: str) -> None:
        """任务进行中：自身内存涨太猛就中止这一单，保住进程。

        按"比开工时多占了多少"算，不按绝对值——模型常驻的那几百 MB 是固定成本，
        不该算进单个任务的预算里。
        """
        rss = read_rss_mb()
        growth = rss - self.started_rss_mb
        if growth > self.limits.max_rss_mb:
            raise ResourceShortage(
                f"本任务让内存多占了 {growth}MB（上限 {self.limits.max_rss_mb}MB，"
                f"当前 {rss}MB，阶段={stage}），已主动中止以免整机 OOM；"
                f"请缩小页范围或降低 DPI")
        mem = read_memory()
        if mem.available_mb < self.limits.min_free_mb // 2:
            raise ResourceShortage(
                f"系统可用内存跌到 {mem.available_mb}MB（阶段={stage}），"
                f"已主动中止本次任务")


def snapshot(cfg: dict | None = None) -> dict:
    """健康检查用：一次返回内存、显存、自身占用。"""
    mem = read_memory()
    gpu = read_gpu()
    data = {"memory": mem.as_dict(), "rss_mb": read_rss_mb(),
            "gpu": gpu.as_dict() if gpu else None}
    if cfg:
        limits = Limits.from_config(cfg)
        data["limits"] = {"min_free_mb": limits.min_free_mb,
                          "max_rss_mb": limits.max_rss_mb,
                          "max_page_pixels": limits.max_page_pixels,
                          "max_pages": limits.max_pages}
        data["ok"] = mem.available_mb >= limits.min_free_mb
    return data


def _print_line(cfg: dict | None) -> None:
    data = snapshot(cfg)
    mem = data["memory"]
    gpu = data["gpu"]
    gpu_text = f"{gpu['name']} 空闲 {gpu['free_mb']}MB/{gpu['total_mb']}MB" if gpu else "无 GPU"
    print(f"内存 可用 {mem['available_mb']}MB/{mem['total_mb']}MB  本进程 {data['rss_mb']}MB"
          f"  |  {gpu_text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="资源预算检查")
    ap.add_argument("--loop", type=float, default=0, help="每 N 秒打一行，0 表示只打一次")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import load_config                    # noqa: E402
    cfg = load_config(args.config)
    if not args.loop:
        _print_line(cfg)
        return 0
    try:
        while True:
            _print_line(cfg)
            time.sleep(args.loop)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
