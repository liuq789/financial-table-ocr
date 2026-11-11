"""网页服务用例：起服务 → 上传图片 / PDF → 拿 JSON + Excel。

不连外网，只用 localhost；legacy 后端跑一张真实页约 2~4s。

    PYTHONPATH=src $PY tests/check_web.py
"""

from __future__ import annotations

import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import web.server as ws                      # noqa: E402

IMAGE = ROOT / "data" / "images" / "ruled" / "000651_p029_asset_change.png"
PDF = ROOT / "data" / "pdfs" / "000651_2023_annual.pdf"

TABLE_HTML = ("<table><tr><td rowspan='2'>项目</td><td colspan='2'>2023 年度</td></tr>"
              "<tr><td>金额</td><td>占比</td></tr>"
              "<tr><td>货币资金</td><td>1,000.00</td><td>25.3%</td></tr></table>")


class FakeModel(BaseHTTPRequestHandler):
    """假 PaddleOCR-VL：只为验证网页 → pipeline → 下载这条链路。"""

    def do_POST(self) -> None:                   # noqa: N802  http.server 约定
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        data = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": TABLE_HTML}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def multipart(fields: dict, name: str, blob: bytes,
              ctype: str) -> tuple[bytes, str]:
    boundary = "----focr" + uuid.uuid4().hex
    out = []
    for key, value in fields.items():
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                   f'name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    out.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f'filename="{name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode("utf-8"))
    out.append(blob)
    out.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def post(base: str, body: bytes, ctype: str) -> tuple[int, dict]:
    req = Request(urljoin(base, "api/run"), data=body, method="POST",
                  headers={"Content-Type": ctype})
    try:
        with urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:                 # 4xx/5xx 也要能拿到 body
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def get(base: str, path: str) -> tuple[int, bytes, str]:
    with urlopen(urljoin(base, path), timeout=60) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


def main() -> int:
    ws.setup_logging("INFO")
    # 服务端现在是有状态的（准入控制 + 限额），必须按真实启动流程初始化
    settings = ws.Settings.from_config(ws.load_config())
    ws.APP = ws.App(settings)
    ws.WORK.mkdir(parents=True, exist_ok=True)
    server = ws.BoundedServer(("127.0.0.1", 0), ws.Handler, settings.max_conns)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"

    code, html, ctype = get(base, "")
    assert code == 200 and "text/html" in ctype, (code, ctype)
    assert b"\xe8\xb4\xa2\xe6\x8a\xa5" in html or b"<html" in html
    print("  ok  首页")

    code, body, _ = get(base, "api/config")
    cfg = json.loads(body.decode("utf-8"))
    assert code == 200 and "model" in cfg and "backends" in cfg, cfg
    assert "api_key" not in cfg["model"], "不该把 api_key 回给浏览器"
    model_name = cfg["model"].get("model")
    print(f"  ok  配置接口（backend={cfg['backend']}，模型={model_name!r}）")

    body, ctype = multipart({"backend": "legacy", "dpi": "200"}, IMAGE.name,
                            IMAGE.read_bytes(), "image/png")
    code, data = post(base, body, ctype)
    assert code == 200, data
    page = data["pages"][0]
    assert page["table_count"] >= 1, page
    assert page["thumb"].startswith("data:image/jpeg;base64,"), "缺缩略图"
    assert page["timings"].get("total", 0) > 0, f"缺分阶段耗时：{page.get('timings')}"
    assert data["timings"].get("total", 0) > 0, f"缺总耗时：{data.get('timings')}"
    print(f"  ok  图片：{page['table_count']} 张表，"
          f"{sum(t['cells'] for t in page['tables'])} 格，{data['elapsed']}s"
          f"（layout={page['timings'].get('layout')}ms "
          f"ocr={page['timings'].get('ocr')}ms）")

    for key, suffix in (("json", ".json"), ("xlsx", ".xlsx")):
        assert data["outputs"][key].endswith(IMAGE.stem + suffix), data["outputs"][key]
        code, blob, ctype = get(base, data["outputs"][key].lstrip("/"))
        assert code == 200 and len(blob) > 200, (key, code, len(blob))
        print(f"  ok  下载 {key}（{len(blob) // 1024}KB，{ctype.split(';')[0]}）")

    # 前端两个"完整查看"接口
    code, blob, _ = get(base, data["outputs"]["doc"].lstrip("/"))
    doc = json.loads(blob)
    assert code == 200 and doc["pages"][0]["page"] == 1
    print(f"  ok  完整 JSON（{len(blob) // 1024}KB，{len(doc['pages'])} 页）")

    code, blob, _ = get(base, data["outputs"]["sheets"].lstrip("/"))
    sheets = json.loads(blob)["sheets"]
    assert code == 200 and sheets, "没有 sheet"
    assert sheets[0]["rows"] and sheets[0]["merges"], sheets[0]
    assert len(sheets[0]["widths"]) >= len(sheets[0]["rows"][0])
    # "要显示全"：正常页不允许截断
    assert not sheets[0]["truncated"], "正常页面不该被截断"

    # 摘要接口（刷新/分享后重看结果）
    code, blob, _ = get(base, "api/summary/" + data["job"])
    summary = json.loads(blob)
    assert code == 200 and summary["job"] == data["job"], (code, summary)
    assert summary["outputs"]["doc"].endswith(data["job"]), summary["outputs"]
    print(f"  ok  任务摘要可重开（{len(summary['pages'])} 页）")
    print(f"  ok  Excel 网格：{len(sheets)} 个 sheet，首表 {len(sheets[0]['rows'])} 行，"
          f"{len(sheets[0]['merges'])} 处合并")

    try:
        get(base, "api/sheets/nosuchjob123")
        raise AssertionError("不存在的任务应该 404")
    except HTTPError as exc:
        assert exc.code == 404
    print("  ok  不存在的任务返回 404")

    body, ctype = multipart({"backend": "legacy", "pages": "29"}, PDF.name,
                            PDF.read_bytes(), "application/pdf")
    code, data = post(base, body, ctype)
    assert code == 200, data
    assert data["pages"][0]["page"] == 29, data["pages"][0]
    assert data["pages"][0]["table_count"] >= 1
    print(f"  ok  PDF 指定页：第 29 页，{data['elapsed']}s")

    body, ctype = multipart({}, "note.txt", b"hello", "text/plain")
    code, data = post(base, body, ctype)
    assert code == 400 and "不支持" in data["error"], (code, data)
    print("  ok  拒绝不支持的类型")

    fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    body, ctype = multipart(
        {"backend": "paddleocr_vl", "model": "test-model",
         "base_url": f"http://127.0.0.1:{fake.server_address[1]}/v1"},
        IMAGE.name, IMAGE.read_bytes(), "image/png")
    code, data = post(base, body, ctype)
    assert code == 200, data
    table = data["pages"][0]["tables"][0]
    assert table["kind"] == "paddleocr_vl", table
    assert data["engine"]["kinds"] == {"paddleocr_vl": 1}, data["engine"]
    assert data["engine"]["fallback_count"] == 0, data["engine"]
    code, blob, _ = get(base, data["outputs"]["xlsx"].lstrip("/"))
    assert code == 200 and len(blob) > 200
    print(f"  ok  PaddleOCR-VL 后端：{table['rows']}×{table['cols']}，{table['cells']} 格")
    fake.shutdown()

    body, ctype = multipart({"backend": "paddleocr_vl", "model": ""}, IMAGE.name,
                            IMAGE.read_bytes(), "image/png")
    body, ctype = multipart({"backend": "paddleocr_vl", "model": "test-model",
                             "base_url": "http://127.0.0.1:59999/v1"},
                            IMAGE.name, IMAGE.read_bytes(), "image/png")
    code, data = post(base, body, ctype)
    assert code == 200, data
    assert data["engine"]["fallback_count"] == 1, data["engine"]
    assert "连接被拒绝" in data["engine"]["fallbacks"][0]["reason"]
    assert "wired" in data["engine"]["kinds"] or "wireless" in data["engine"]["kinds"]
    print(f"  ok  模型不可用时如实报退回（{data['engine']['kinds']}）")

    body, ctype = multipart({"backend": "paddleocr_vl", "model": ""}, IMAGE.name,
                            IMAGE.read_bytes(), "image/png")
    code, data = post(base, body, ctype)
    assert code >= 400 and "模型名" in data.get("error", ""), (code, data.get("error"))
    print("  ok  未配模型名时报错清楚")

    server.shutdown()
    print("13 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
