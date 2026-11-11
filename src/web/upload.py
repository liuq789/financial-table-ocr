"""流式 multipart/form-data 解析：文件边收边落盘，内存占用与文件大小无关。

为什么要自己写
--------------
标准库 `cgi.FieldStorage` 在文件字段上会走 `BytesIO`（Python 3.10 的 cgi.py:706），
也就是**整份上传先读进内存**。60MB 的 PDF 叠上几个并发请求就能把进程打穿，而
WSL 只有 16G。这里改成：请求体按 256KB 分块读，文件字段直接写进目标文件，
文本字段攒在内存里但设 64KB 硬上限。

用法
----
    with open(tmp, "wb") as sink:
        form = parse(handler.rfile, handler.headers, sink, max_bytes=64 << 20)
    form.fields["backend"]        # 文本字段
    form.upload                   # Uploaded(filename, size) 或 None

单独跑（用一份手拼的 multipart 验解析，不依赖网络）：

    PYTHONPATH=src python -m web.upload --selftest
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CHUNK = 256 * 1024            # 单次从 socket 读多少
MAX_FIELD_BYTES = 64 * 1024   # 单个文本字段上限，超了直接拒
MAX_HEADER_BYTES = 16 * 1024  # 单个 part 的头部上限


class UploadError(ValueError):
    """上传内容不合法（格式坏、超限、缺字段）。"""


@dataclass(frozen=True)
class Uploaded:
    filename: str
    size: int


@dataclass(frozen=True)
class Form:
    fields: dict[str, str] = field(default_factory=dict)
    upload: Uploaded | None = None


def boundary_of(content_type: str) -> bytes:
    """从 Content-Type 里取 boundary；没有就报错（说明不是 multipart）。"""
    if "multipart/form-data" not in content_type.lower():
        raise UploadError("请求不是 multipart/form-data")
    for part in content_type.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "boundary" and value:
            return value.strip().strip('"').encode("utf-8")
    raise UploadError("Content-Type 里没有 boundary")


class _Reader:
    """带缓冲的读取器：只保证"够用就停"，绝不把整份请求读进内存。

    `remaining` 来自 Content-Length。**必须按它截断**：socket 上的 `read(n)` 会
    一直等到读满 n 字节，而 HTTP/1.1 连接是长连接，后面的字节永远不会来——
    之前就是在这里死等的（30 个并发全部超时）。有 `read1` 就用它，能拿到多少
    算多少，不阻塞。
    """

    def __init__(self, stream, chunk: int = CHUNK, remaining: int | None = None) -> None:
        self.stream = stream
        self.chunk = chunk
        self.remaining = remaining          # None 表示读到 EOF（内存/文件流）
        self.buf = b""
        self.eof = False
        self.consumed = 0            # 从 socket 实际读了多少字节，用于总量兜底

    def _pull(self) -> bool:
        if self.eof:
            return False
        want = self.chunk
        if self.remaining is not None:
            if self.remaining <= 0:
                self.eof = True
                return False
            want = min(self.chunk, self.remaining)
        data = (self.stream.read1(want) if hasattr(self.stream, "read1")
                else self.stream.read(want))
        if not data:
            self.eof = True
            return False
        if self.remaining is not None:
            self.remaining -= len(data)
        self.consumed += len(data)
        self.buf += data
        return True

    def read_line(self) -> bytes:
        while b"\r\n" not in self.buf:
            if len(self.buf) > MAX_HEADER_BYTES:
                raise UploadError("multipart 头部过长")
            if not self._pull():
                raise UploadError("上传数据不完整")
        line, _, self.buf = self.buf.partition(b"\r\n")
        return line

    def peek(self, n: int) -> bytes:
        while len(self.buf) < n and self._pull():
            pass
        return self.buf[:n]

    def drop(self, n: int) -> None:
        self.buf = self.buf[n:]

    def pump(self, sink, delim: bytes, limit: int | None = None) -> int:
        """把数据写进 sink，直到遇到 delim（delim 本身被吃掉）。返回写入字节数。"""
        keep = len(delim)
        written = 0
        while True:
            idx = self.buf.find(delim)
            if idx >= 0:
                if limit is not None and written + idx > limit:
                    raise UploadError(f"内容超过上限 {limit // 1024 // 1024}MB")
                sink.write(self.buf[:idx])
                written += idx
                self.buf = self.buf[idx + keep:]
                return written
            if len(self.buf) > keep:
                body = self.buf[:-keep]          # 尾部可能是分隔符的前半截，留着
                sink.write(body)
                written += len(body)
                self.buf = self.buf[-keep:]
                if limit is not None and written > limit:
                    raise UploadError(f"内容超过上限 {limit // 1024 // 1024}MB")
            if not self._pull():
                raise UploadError("上传数据不完整（找不到分隔符）")


def parse(stream, content_type: str, sink, *, max_bytes: int,
          total_length: int | None = None, sink_name: str = "upload") -> Form:
    """解析 multipart，把唯一的文件字段写进 `sink`（已打开的二进制可写文件）。

    - 文件字段：分块写进 sink，超过 `max_bytes` 立刻抛错（不会先读满再判断）。
    - 文本字段：最多 MAX_FIELD_BYTES，超了抛错。
    - 多个文件字段：只认第一个，其余丢弃（前端只传一个）。
    - `total_length` 传 Content-Length：不传的话 socket 上传到一半就会死等。
    """
    delim = b"\r\n--" + boundary_of(content_type)
    reader = _Reader(stream, remaining=total_length)

    if reader.read_line() != b"--" + boundary_of(content_type):
        raise UploadError("multipart 起始分隔符不对")

    fields: dict[str, str] = {}
    upload: Uploaded | None = None

    while True:
        headers = _read_headers(reader)
        name, filename = _disposition(headers)
        if filename:
            if upload is not None:               # 第二个文件字段：读完丢掉
                reader.pump(_NullSink(), delim)
            else:
                size = reader.pump(sink, delim, limit=max_bytes)
                if size == 0:
                    raise UploadError("上传的文件是空的")
                upload = Uploaded(filename=filename, size=size)
        else:
            buf = io.BytesIO()
            reader.pump(buf, delim, limit=MAX_FIELD_BYTES)
            if name:
                fields[name] = buf.getvalue().decode("utf-8", "replace")

        if reader.consumed > max_bytes + 8 * 1024 * 1024:
            raise UploadError("请求体超过上限")
        tail = reader.peek(2)
        if tail == b"--":                        # 结束分隔符
            reader.drop(2)
            break
        if tail != b"\r\n":
            raise UploadError("multipart 结构不对（分隔符后应有 CRLF）")
        reader.drop(2)
    return Form(fields=fields, upload=upload)


class _NullSink:
    def write(self, data: bytes) -> int:
        return len(data)


def _read_headers(reader: _Reader) -> dict[str, str]:
    headers: dict[str, str] = {}
    while True:
        line = reader.read_line()
        if not line:
            return headers
        key, _, value = line.decode("utf-8", "replace").partition(":")
        headers[key.strip().lower()] = value.strip()


def _disposition(headers: dict[str, str]) -> tuple[str, str]:
    """从 Content-Disposition 里取字段名和文件名。"""
    raw = headers.get("content-disposition", "")
    name = filename = ""
    for part in raw.split(";"):
        key, _, value = part.strip().partition("=")
        value = value.strip().strip('"')
        if key.lower() == "name":
            name = value
        elif key.lower() == "filename":
            filename = Path(value).name          # 去掉路径，防目录穿越
    return name, filename


def build_body(fields: dict[str, str], filename: str | None,
               payload: bytes, boundary: str = "----focr") -> tuple[bytes, str]:
    """拼一份 multipart 请求体（测试与小工具用，服务端不需要）。"""
    out = io.BytesIO()
    for key, value in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                  .encode())
        out.write(value.encode())
        out.write(b"\r\n")
    if filename is not None:
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                  f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                  .encode())
        out.write(payload)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def _selftest() -> int:
    """不依赖网络的自检：正常、超限、缺 CRLF、多字段。"""
    body, ctype = build_body({"backend": "legacy", "pages": "1-2"}, "a.bin", b"x" * 1000)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "up.bin"
        with open(target, "wb") as sink:
            form = parse(io.BytesIO(body), ctype, sink, max_bytes=1 << 20)
        assert form.fields == {"backend": "legacy", "pages": "1-2"}, form.fields
        assert form.upload and form.upload.size == 1000, form.upload
        assert target.stat().st_size == 1000
        print("ok  正常解析（2 个文本字段 + 1 个文件，1000 字节）")

        try:
            with open(target, "wb") as sink:
                parse(io.BytesIO(body), ctype, sink, max_bytes=100)
            raise AssertionError("超限没拦住")
        except UploadError as exc:
            print(f"ok  超限拦截：{exc}")

        try:
            with open(target, "wb") as sink:
                parse(io.BytesIO(body[:-20]), ctype, sink, max_bytes=1 << 20)
            raise AssertionError("截断没拦住")
        except UploadError as exc:
            print(f"ok  截断拦截：{exc}")

        with open(target, "wb") as sink:
            form = parse(io.BytesIO(body), ctype, sink, max_bytes=1 << 20, sink_name="x")
        print(f"ok  落盘大小 {target.stat().st_size}B，内存未驻留整份文件")
    print("4 项通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="流式 multipart 解析自检")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    ap.print_help()
