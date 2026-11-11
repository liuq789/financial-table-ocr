"""paddleocr_vl 分支的离线集成用例：起一个假的 OpenAI 兼容服务，跑通整页。

不依赖 GPU 和真模型，只验证"选路 → 请求 → 解析 → 分区 → 网格完整"这条线。
真服务对拍用 `tests/test_paddleocr_vl.py --live <url> --image <裁图>`。

    PYTHONPATH=src $PY tests/check_paddleocr_branch.py
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config                 # noqa: E402
from pipeline import Pipeline                  # noqa: E402

IMAGE = ROOT / "data" / "images" / "ruled" / "000651_p029_asset_change.png"
HTML = ("<table>"
        "<tr><td rowspan='2'>项目</td><td colspan='2'>2023 年度</td></tr>"
        "<tr><td>金额</td><td>占比</td></tr>"
        "<tr><td>货币资金</td><td>1,000.00</td><td>25.3%</td></tr>"
        "<tr><td>应收账款</td><td>500.00</td><td>12.6%</td></tr>"
        "</table>")


class _Handler(BaseHTTPRequestHandler):
    def _send(self, content: str) -> None:
        data = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        parts = body["messages"][0]["content"]
        assert any(p.get("type") == "image_url" for p in parts), "请求里没有图片"
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        self._send(f"```html\n{HTML}\n```")

    def log_message(self, *args) -> None:
        pass


class _RepairHandler(_Handler):
    """第一次返回废话，第二次返回合法表格：验证"不合格就换提示词重问"。"""

    calls = 0

    def do_POST(self) -> None:
        type(self).calls += 1
        if type(self).calls == 1:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self._send("抱歉，我无法识别这张图片里的表格。")
            return
        super().do_POST()


def _serve(handler: type[BaseHTTPRequestHandler] = _Handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _table_of(page: dict) -> dict:
    tables = [r for r in page["regions"] if r["type"] == "table"]
    assert len(tables) == 1, f"应该只有 1 张表，实际 {len(tables)}"
    return tables[0]


def _holes(table: dict) -> int:
    cover = [[0] * table["n_cols"] for _ in range(table["n_rows"])]
    for c in table["cells"]:
        for dr in range(c["rowspan"]):
            for dc in range(c["colspan"]):
                cover[c["row"] + dr][c["col"] + dc] += 1
    return sum(1 for row in cover for v in row if v == 0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    image = cv2.imread(str(IMAGE))
    assert image is not None, f"读不了 {IMAGE}"
    server = _serve()
    port = server.server_address[1]

    cfg = load_config()
    cfg["tablerec"]["backend"] = "paddleocr_vl"
    cfg["tablerec"]["paddleocr_vl"].update({"base_url": f"http://127.0.0.1:{port}/v1",
                                   "model": "test-model", "repair_attempts": 1})
    page, _ = Pipeline(cfg).process_page(image, 1, do_preprocess=False)

    table = _table_of(page)
    assert table["kind"] == "paddleocr_vl", table["kind"]
    assert (table["n_rows"], table["n_cols"]) == (4, 3), (table["n_rows"], table["n_cols"])
    assert len(table["cells"]) == 10, len(table["cells"])
    assert _holes(table) == 0, f"有洞 {_holes(table)}"
    head = {(c["row"], c["col"]): c for c in table["cells"]}
    assert head[(0, 0)]["rowspan"] == 2 and head[(0, 1)]["colspan"] == 2
    assert head[(2, 1)]["ocr"] == "1,000.00", head[(2, 1)]["ocr"]
    assert head[(2, 1)]["bbox"][:2] != [0, 0], "bbox 没加上表格偏移"
    print(f"  ok  paddleocr_vl 分支：{table['n_rows']}×{table['n_cols']}，"
          f"{len(table['cells'])} 格，洞=0")

    # 第一次输出认不出 → 换提示词重问，第二次拿到合法表格
    server.shutdown()
    server = _serve(_RepairHandler)
    cfg["tablerec"]["paddleocr_vl"]["base_url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    page, _ = Pipeline(cfg).process_page(image, 1, do_preprocess=False)
    table = _table_of(page)
    assert table["kind"] == "paddleocr_vl", table["kind"]
    assert _RepairHandler.calls == 2, f"应该重问一次，实际 {_RepairHandler.calls} 次"
    print(f"  ok  重问：首次输出不可解析，第二次拿到 {table['n_rows']}×{table['n_cols']}")

    # 服务不可达 + fallback_legacy=true → 退回本地链路，不能整页失败
    server.shutdown()
    cfg["tablerec"]["paddleocr_vl"]["retries"] = 0
    page, _ = Pipeline(cfg).process_page(image, 1, do_preprocess=False)
    table = _table_of(page)
    assert table["kind"] in ("wired", "wireless"), table["kind"]
    print(f"  ok  兜底：服务不可达退回 legacy（{table['kind']}，"
          f"{len(table['cells'])} 格）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
