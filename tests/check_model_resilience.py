"""模型侧加固验证：PaddleOCR-VL 调用在任何异常下都不能把系统拖垮。

覆盖五件事：
1. 服务端报"上下文超长" → 自动缩小重发，最终成功（不是直接失败）
2. 服务端 5xx → 按 retries 重试，最终成功
3. 服务端卡死 → 受 total_timeout / timeout 约束，准时放弃（不无限等）
4. 送图尺寸受 max_side / max_pixels 约束（大图不会把上下文顶爆）
5. 模型彻底不可用 → 整页仍然出结果（退回本地链路），进程不受影响

    PYTHONPATH=src python tests/check_model_resilience.py

不需要 GPU、不需要真模型：起一个假的 OpenAI 兼容服务。
"""

from __future__ import annotations

import base64
import io
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.paddleocr_vl import PaddleOcrVl, PaddleOcrVlError   # noqa: E402

HTML = """<table>
<tr><td>项目</td><td>2023 年</td></tr>
<tr><td>货币资金</td><td>1,234.00</td></tr>
</table>"""


class FakeModel(BaseHTTPRequestHandler):
    """按脚本回答问题：script 里每个元素决定这一次的行为。"""

    script: list[str] = []
    max_model_len = 16384
    seen: list[dict] = []
    hang_s = 0.0

    def do_GET(self) -> None:                     # noqa: N802
        if self.path.startswith("/v1/models"):
            body = json.dumps({"object": "list", "data": [
                {"id": "PaddleOCR-VL", "object": "model",
                 "max_model_len": self.max_model_len}]}).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"{}", "application/json")

    def do_POST(self) -> None:                    # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        image = payload["messages"][0]["content"][0]["image_url"]["url"]
        raw = base64.b64decode(image.split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.__class__.seen.append({
            "size": (decoded.shape[1], decoded.shape[0]) if decoded is not None else (0, 0),
            "max_tokens": payload.get("max_tokens"),
            "prompt": payload["messages"][0]["content"][1]["text"][:20],
        })
        step = self.script.pop(0) if self.script else "ok"
        if self.hang_s:
            time.sleep(self.hang_s)
        if step == "overflow":
            self._send(400, json.dumps({"error": {
                "message": "This model's maximum context length is 16384 tokens. "
                           "However, you requested 21000 tokens."}}).encode(), "application/json")
            return
        if step == "500":
            self._send(500, b'{"error": "internal"}', "application/json")
            return
        body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                   "content": HTML}}],
                           "usage": {"prompt_tokens": 900, "completion_tokens": 120}}).encode()
        self._send(200, body, "application/json")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def start(script: list[str], max_model_len: int = 16384, hang_s: float = 0.0):
    FakeModel.script = list(script)
    FakeModel.seen = []
    FakeModel.max_model_len = max_model_len
    FakeModel.hang_s = hang_s
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/v1"


def cfg(url: str, **over) -> dict:
    base = {"base_url": url, "model": "PaddleOCR-VL", "api_key": "EMPTY",
            "prompt": "Table Recognition:", "max_tokens": 8192, "temperature": 0.0,
            "timeout": 20, "retries": 2, "max_side": 1600, "max_pixels": 2_560_000,
            "total_timeout": 60, "shrink_attempts": 2, "repair_attempts": 1}
    base.update(over)
    return base


def image(width: int = 1200, height: int = 800) -> np.ndarray:
    """白底黑字的假表格图，够大才看得出缩图效果。"""
    img = np.full((height, width, 3), 255, np.uint8)
    for y in range(40, height, 40):
        cv2.line(img, (20, y), (width - 20, y), (0, 0, 0), 1)
    return img


def check_overflow_shrinks() -> None:
    server, url = start(["overflow", "ok"])
    try:
        result = PaddleOcrVl(cfg(url)).predict(image())
        assert result.n_rows == 2, result
        assert len(FakeModel.seen) == 2, FakeModel.seen
        first, second = FakeModel.seen[0]["size"], FakeModel.seen[1]["size"]
        assert second[0] < first[0] and second[1] < first[1], (first, second)
        print(f"  ok  上下文超长自动缩图：{first[0]}×{first[1]} → {second[0]}×{second[1]}，最终成功")
    finally:
        server.shutdown()


def check_retry_5xx() -> None:
    server, url = start(["500", "500", "ok"])
    try:
        result = PaddleOcrVl(cfg(url)).predict(image(600, 400))
        assert result.n_rows == 2, result
        assert len(FakeModel.seen) == 3, len(FakeModel.seen)
        print(f"  ok  两次 5xx 后重试成功（共 {len(FakeModel.seen)} 次请求）")
    finally:
        server.shutdown()


def check_hang_bounded() -> None:
    server, url = start([], hang_s=30)
    try:
        started = time.perf_counter()
        try:
            PaddleOcrVl(cfg(url, timeout=1, retries=1, total_timeout=3)).predict(image(400, 300))
            raise AssertionError("卡死时没有放弃")
        except PaddleOcrVlError as exc:
            cost = time.perf_counter() - started
            assert cost < 8, f"释放太慢：{cost:.1f}s"
            print(f"  ok  服务卡死时 {cost:.1f}s 内放弃（预算 3s）：{str(exc)[:46]}…")
    finally:
        server.shutdown()


def check_image_limits() -> None:
    server, url = start(["ok"])
    try:
        big = image(6000, 4000)                   # 24MP，远超上限
        PaddleOcrVl(cfg(url)).predict(big)
        w, h = FakeModel.seen[0]["size"]
        assert max(w, h) <= 1600, (w, h)
        assert w * h <= 2_560_000, (w, h, w * h)
        print(f"  ok  24MP 大图先缩到 {w}×{h}（≤1600 长边 / ≤2.56MP）才发出去")
    finally:
        server.shutdown()


def check_token_budget() -> None:
    """服务端上下文只有 4096 → max_tokens 必须被压下来，不能照发 8192。"""
    server, url = start(["ok"], max_model_len=4096)
    try:
        PaddleOcrVl(cfg(url)).predict(image(800, 600))
        sent = FakeModel.seen[0]["max_tokens"]
        assert sent is not None and sent <= 4096, sent
        print(f"  ok  上下文 4096 时 max_tokens 压到 {sent}（配置写的是 8192）")
    finally:
        server.shutdown()


def check_pipeline_survives() -> None:
    """模型彻底不可用：整页仍要出结果（退回 legacy），不能抛到用户脸上。"""
    from config import load_config
    from pipeline import Pipeline

    server, url = start(["500"] * 50)             # 永远失败
    try:
        config = load_config()
        config["tablerec"]["backend"] = "paddleocr_vl"
        config["tablerec"]["paddleocr_vl"].update(cfg(url, retries=0, timeout=3))
        pipe = Pipeline(config)
        page = ROOT / "data" / "images" / "ruled" / "000651_p029_asset_change.png"
        result, _ = pipe.process_page(cv2.imread(str(page)), 1, do_preprocess=False)
        tables = [r for r in result["regions"] if r["type"] == "table"]
        assert tables, "模型挂了整页就没表了"
        assert tables[0]["kind"] in ("wired", "wireless"), tables[0]["kind"]
        assert result["fallbacks"], "没有记录降级原因"
        print(f"  ok  模型全挂时整页仍有 {len(tables)} 张表（{tables[0]['kind']}，"
              f"降级 {len(result['fallbacks'])} 次）")
    finally:
        server.shutdown()


def main() -> int:
    print("== 模型侧容错")
    check_overflow_shrinks()
    check_retry_5xx()
    check_hang_bounded()
    check_image_limits()
    check_token_budget()
    print("== 整页兜底")
    check_pipeline_survives()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
