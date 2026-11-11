"""加固验证：并发打满时进程不崩、内存不涨、上传不驻留内存、产物自动清理。

这个用例回答一个问题：**系统在最坏情况下会不会挂**。
所以它不只是"跑通"，还断言：
- 30 个并发上传全部得到 200/429/503 之一，**没有一个 500**（500 说明内部崩了）
- 服务进程 PID 没变（没被 OOM 重启），结束后还能正常接活
- 进程 RSS 增长有上限（模型只加载一份，不随请求数增长）
- 超过上限的上传被 413 快速拒绝，且服务继续可用
- 畸形 multipart 被 400 拒绝，且服务继续可用
- 产物清理按 keep_hours / max_work_mb 生效

    PYTHONPATH=src python tests/check_load.py [--jobs 30]

不依赖 GPU，也不需要外部服务（全部走 legacy 本地链路）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE = ROOT / "data" / "images" / "ruled" / "000651_p029_asset_change.png"
PY = sys.executable


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url: str, timeout: float = 15) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def post_file(base: str, path: Path, fields: dict[str, str], timeout: float = 120
              ) -> tuple[int, dict]:
    boundary = "----loadtest"
    body = b""
    for key, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
             ).encode()
    body += path.read_bytes() + b"\r\n" + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(base + "/api/run", data=body, method="POST",
                                 headers={"Content-Type":
                                          f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def oversize(port: int, limit_mb: int) -> int:
    """只发请求头不发正文：服务端应立刻 413，而不是等收完 70MB 再拒。"""
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", "/api/run")
    conn.putheader("Content-Type", "multipart/form-data; boundary=x")
    conn.putheader("Content-Length", str((limit_mb + 10) * 1024 * 1024))
    conn.endheaders()
    resp = conn.getresponse()
    code = resp.status
    conn.close()
    return code


def wait_ready(port: int, timeout: float = 40) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, _ = get(f"http://127.0.0.1:{port}/api/health", timeout=2)
            if code == 200:
                return True
        except OSError:
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------- 单测（不需要服务）
def check_cleanup() -> None:
    """keep_hours 过期删、总量超限按最旧淘汰。"""
    from web.server import Settings, cleanup_jobs

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        old = work / "20200101-000000-dead"
        (old / "pages").mkdir(parents=True)
        (old / "a.json").write_bytes(b"x" * 1024)
        stamp = time.time() - 48 * 3600
        os.utime(old, (stamp, stamp))

        fresh = work / "20260101-000000-live"
        fresh.mkdir()
        (fresh / "b.bin").write_bytes(b"y" * (3 * 1024 * 1024))

        import web.server as ws
        original = ws.WORK
        ws.WORK = work
        try:
            settings = Settings(keep_hours=24, max_work_mb=1, max_conns=4)
            removed = cleanup_jobs(settings, force=True)
            assert not old.exists(), "过期目录没删"
            assert removed >= 1, removed
            # 3MB 超过 1MB 上限 → 连新目录也要被淘汰
            assert not fresh.exists(), "超过总量上限时没按最旧淘汰"
            print(f"  ok  产物清理：删了 {removed} 个（过期 + 超总量）")
        finally:
            ws.WORK = original


def check_guard_limits() -> None:
    """预算：内存不足要抛 ResourceShortage，而不是继续干。"""
    from resource_guard import Budget, Limits, ResourceShortage

    limits = Limits(min_free_mb=10 ** 7, max_rss_mb=10 ** 7, max_page_pixels=100)
    try:
        Budget(limits).precheck()
        raise AssertionError("内存门槛没生效")
    except ResourceShortage as exc:
        print(f"  ok  内存门槛拦截：{str(exc)[:48]}…")

    # 预算按"比开工时多占了多少"算，所以先把基线压低 100MB 来模拟增长
    from resource_guard import read_rss_mb

    budget = Budget(Limits(min_free_mb=0, max_rss_mb=50))
    budget.started_rss_mb = read_rss_mb() - 100
    try:
        budget.check("测试")
        raise AssertionError("RSS 上限没生效")
    except ResourceShortage as exc:
        print(f"  ok  任务内增量上限拦截：{str(exc)[:52]}…")


# ---------------------------------------------------------------- 集成（真起服务）
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    assert IMAGE.exists(), f"缺样本 {IMAGE}"

    print("== 单测")
    check_cleanup()
    check_guard_limits()

    print("== 集成（真起服务，跑 legacy）")
    port = free_port()
    log = ROOT / "work" / "logs" / "loadtest.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.Popen([PY, "-m", "web.server", "--port", str(port),
                             "--log-level", "WARNING", "--log-file", str(log)],
                            cwd=str(ROOT), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        assert wait_ready(port), "服务没起来"
        code, health = get(base + "/api/health")
        print(f"  ok  健康检查：可用内存 {health['memory']['available_mb']}MB，"
              f"本进程 {health['rss_mb']}MB，并发 {health['gate']['max_inflight']}"
              f"+排队 {health['gate']['max_pending']}")

        # 先跑一单把模型加载进来：模型常驻内存是固定成本，不该算成"泄漏"
        code, _ = post_file(base, IMAGE, {"backend": "legacy", "preprocess": "0"})
        assert code == 200, code
        before = get(base + "/api/health")[1]["rss_mb"]
        print(f"  ok  预热完成（模型已加载）：RSS 基线 {before}MB")

        # 30 个并发上传：只允许 200 / 429 / 503
        results: list[tuple[int, str]] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            code, data = post_file(base, IMAGE, {"backend": "legacy", "preprocess": "0"})
            with lock:
                results.append((code, data.get("error", "")[:60]))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.jobs)]
        started = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(240)
        took = time.time() - started
        codes = [c for c, _ in results]
        ok = codes.count(200)
        busy = codes.count(429) + codes.count(503)
        bad = [r for r in results if r[0] not in (200, 429, 503)]
        assert not bad, f"出现了非预期状态：{bad[:3]}"
        assert ok >= 1, f"一个都没成功：{codes}"
        print(f"  ok  {args.jobs} 个并发：成功 {ok}，限流 {busy}，"
              f"无 500，用时 {took:.1f}s")

        # 进程活着、PID 没变、内存没失控
        assert proc.poll() is None, "服务进程死了（被 OOM？）"
        code, health2 = get(base + "/api/health")
        assert code == 200, code
        after = health2["rss_mb"]
        growth = after - before
        assert growth < 600, f"RSS 涨了 {growth}MB，疑似泄漏"
        assert after < 4096, f"RSS 绝对值 {after}MB 过高"
        print(f"  ok  进程存活 pid={proc.pid}，RSS {before}→{after}MB（+{growth}），"
              f"30 请求未累积内存")
        assert health2["gate"]["accepted"] >= 1, health2["gate"]
        assert health2["gate"]["inflight"] == 0, f"有任务没释放：{health2['gate']}"

        # 超限上传：立刻 413，不发正文
        code = oversize(port, 60)
        assert code == 413, f"超限上传返回 {code}，应为 413"
        print("  ok  超限上传 413（只发头就拒，不读 70MB）")

        # 畸形 multipart：400，且服务仍可用
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/api/run", body=b"garbage",
                     headers={"Content-Type": "multipart/form-data; boundary=x"})
        resp = conn.getresponse()
        assert resp.status == 400, resp.status
        conn.close()
        print("  ok  畸形上传 400")

        # 打完这些之后还能正常干活
        code, data = post_file(base, IMAGE, {"backend": "legacy", "preprocess": "0"})
        assert code == 200 and data["pages"][0]["table_count"] >= 1, (code, data)
        print(f"  ok  压力之后仍可正常识别（{data['pages'][0]['table_count']} 张表，"
              f"{data['elapsed']}s）")

        # 任务目录没有无限膨胀
        code, health3 = get(base + "/api/health")
        assert health3["jobs_dir_mb"] < 4096, health3["jobs_dir_mb"]
        print(f"  ok  产物目录 {health3['jobs_dir_mb']}MB（受 max_work_mb 约束）")
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(ROOT / "work" / "web", ignore_errors=True)

    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
