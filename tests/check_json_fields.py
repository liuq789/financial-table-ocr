"""确认接口输出里 char 分区仍然带 lines，并打印一个样例。"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data" / "images" / "ruled" / "000651_p029_asset_change.png"
BASE = "http://127.0.0.1:8010"


def post(path: str, image: Path) -> dict:
    boundary = "----check"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"backend\"\r\n\r\nlegacy\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{image.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode() + image.read_bytes() + b"\r\n" + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                 headers={"Content-Type":
                                          f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main() -> int:
    data = post("/api/run", IMAGE)
    with urllib.request.urlopen(f"{BASE}/api/doc/{data['job']}", timeout=30) as resp:
        doc = json.loads(resp.read())
    chars = [r for r in doc["pages"][0]["regions"] if r["type"] == "char"]
    with_lines = sum(1 for c in chars if c.get("lines"))
    print(f"char 分区 {len(chars)} 个，带 lines 的 {with_lines} 个")
    print("样例：" + json.dumps(chars[0]["lines"][0], ensure_ascii=False))
    table = [r for r in doc["pages"][0]["regions"] if r["type"] == "table"][0]
    print(f"表格 {table['n_rows']}×{table['n_cols']}，{len(table['cells'])} 格")
    assert with_lines == len(chars), "有 char 分区丢了 lines"
    print("ok  lines 保留完好")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
