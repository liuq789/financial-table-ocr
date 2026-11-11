"""准入控制：把"无限并发"变成"1 个在跑 + N 个排队 + 其余立刻被拒"。

为什么必须有
------------
ThreadingHTTPServer 来一个连接开一个线程，没有上限。几十个并发上传进来就是
几十份 60MB 文件 + 几十个 ONNX 推理同时抢内存，进程必死。这里做三件事：

1. **在跑的并发有上限**（默认 1：ONNX/GPU 本来也不能并行）。
2. **排队有上限**（默认 4）：满了直接拒，而不是让请求无限堆着把内存/线程吃光。
3. **排队有超时**：等太久就告诉客户端"忙"，别让它挂着心跳超时。

被拒是 429/503 + `Retry-After`，服务本身不受影响——这是"永不失败"的实现方式。

单独跑
------
    PYTHONPATH=src python -m web.gate --selftest
"""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass


class Busy(RuntimeError):
    """排队已满或等待超时。上层转 429，服务继续工作。"""


@dataclass(frozen=True)
class GateStats:
    inflight: int
    pending: int
    max_inflight: int
    max_pending: int
    accepted: int
    rejected: int
    active_job: str
    longest_wait_s: float


class JobGate:
    """有界并发 + 有界排队。线程安全。"""

    def __init__(self, max_inflight: int = 1, max_pending: int = 4,
                 wait_timeout: float = 30.0) -> None:
        self._sem = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._inflight = 0
        self._pending = 0
        self._accepted = 0
        self._rejected = 0
        self._active = ""
        self._longest_wait = 0.0
        self.max_inflight = max_inflight
        self.max_pending = max_pending
        self.wait_timeout = wait_timeout

    @contextmanager
    def slot(self, job_id: str = ""):
        """拿到执行位才进 with 体；拿不到抛 Busy。"""
        started = time.monotonic()
        if not self._sem.acquire(blocking=False):
            with self._lock:
                if self._pending >= self.max_pending:
                    self._rejected += 1
                    raise Busy(f"服务忙：已有 {self.max_pending} 个任务在排队，请稍后重试")
                self._pending += 1
            try:
                if not self._sem.acquire(timeout=self.wait_timeout):
                    with self._lock:
                        self._rejected += 1
                    raise Busy(f"排队超过 {self.wait_timeout:.0f}s 仍未轮到，请稍后重试")
            finally:
                with self._lock:
                    self._pending -= 1
        waited = time.monotonic() - started
        with self._lock:
            self._inflight += 1
            self._accepted += 1
            self._active = job_id
            self._longest_wait = max(self._longest_wait, waited)
        try:
            yield waited
        finally:
            with self._lock:
                self._inflight -= 1
                self._active = ""
            self._sem.release()

    def stats(self) -> GateStats:
        with self._lock:
            return GateStats(inflight=self._inflight, pending=self._pending,
                             max_inflight=self.max_inflight, max_pending=self.max_pending,
                             accepted=self._accepted, rejected=self._rejected,
                             active_job=self._active, longest_wait_s=round(self._longest_wait, 2))


def _selftest() -> int:
    """并发打满 → 排队满 → 拒绝；结束后计数正确。"""
    gate = JobGate(max_inflight=1, max_pending=1, wait_timeout=0.2)
    results: list[str] = []
    release = threading.Event()

    def worker(tag: str) -> None:
        try:
            with gate.slot(tag):
                results.append(f"{tag}:run")
                if tag == "a":
                    release.wait(2)
        except Busy as exc:
            results.append(f"{tag}:busy({exc})")

    first = threading.Thread(target=worker, args=("a",))
    first.start()
    time.sleep(0.05)
    second = threading.Thread(target=worker, args=("b",))     # 排队
    second.start()
    time.sleep(0.05)
    third = threading.Thread(target=worker, args=("c",))      # 排队已满 → 拒
    third.start()
    for thread in (first, second, third):
        thread.join(3)
    release.set()
    second.join(3)

    assert any(r.startswith("a:run") for r in results), results
    assert any(r.startswith("c:busy") for r in results), results
    st = gate.stats()
    assert st.inflight == 0 and st.pending == 0, st
    assert st.rejected >= 1, st
    print(f"ok  并发 1 / 排队 1：{results}")
    print(f"ok  计数 inflight={st.inflight} pending={st.pending} "
          f"accepted={st.accepted} rejected={st.rejected}")
    print("2 项通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="准入控制自检")
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(_selftest())
    ap.print_help()
