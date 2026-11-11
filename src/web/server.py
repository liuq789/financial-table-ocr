"""上传图片 / PDF → 返回 json + excel 的本地网页服务。

只依赖标准库（http.server + 自带的流式 multipart 解析），不引 FastAPI/Flask：
这东西只是给 pipeline 套一层薄壳，装机即用。

    cd financial_ocr
    PYTHONPATH=src python -m web.server --port 8010
    # 浏览器打开 http://127.0.0.1:8010

接口：
    GET  /                    前端单页
    GET  /api/config          表单默认值（来自 config.yaml，api_key 不返回）
    GET  /api/probe?base_url= 探活视觉大模型服务（/v1/models）
    POST /api/run             上传并识别，返回摘要 + 下载链接
    GET  /api/summary/<job>   任务摘要（刷新/分享后可重新打开结果）
    GET  /api/doc/<job>       完整 JSON（内联，供前端语法高亮）
    GET  /api/sheets/<job>    xlsx 解析成网格（行值 + 合并区 + 列宽），供"像 Excel 一样"展示
    GET  /api/file/<job>/<f>  下载 json / xlsx

资源保护（目标：进程永不被 OOM 杀掉，只允许单个请求失败）
--------------------------------------------------------
1. **上传流式落盘**：`web/upload.py` 按 256KB 分块写文件，内存与文件大小无关
   （标准库 cgi 会把整份上传读进 BytesIO，60MB × 并发 = 直接打穿）。
2. **准入控制**：`web/gate.py` 限定"1 个在跑 + 4 个排队"，满了一律 429，
   不让请求无限堆着占内存和线程。
3. **资源预算**：`resource_guard.Budget` 开工前查 /proc/meminfo 与 nvidia-smi，
   不够就 503；每页之间再查一次，涨太猛主动中止这一单。
4. **Pipeline 缓存有上限**：只留最近一份，防止有人每次改参数把模型加载十几份。
5. **连接数上限**：超过 `guard.max_conns` 直接 503，避免线程表被打穿。
6. **产物自动清理**：按 `keep_hours` 与 `max_work_mb` 淘汰旧任务目录。

其它取舍：
- **推理串行**：ONNX / GPU 不能并行，用准入控制排队；静态页和下载不受影响。
- 任务产物落 `work/web/<job>/`，目录名带时间戳。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import logging
import re
import shutil
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]      # src/web/server.py → 项目根
WEB_DIR = Path(__file__).resolve().parent
for extra in (ROOT, WEB_DIR, ROOT / "src"):      # 三种启动方式（-m / import / 直接跑）都能 import
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from config import load_config                       # noqa: E402
from logging_setup import TOTAL_ORDER, job_ctx, setup_logging   # noqa: E402
from pdf2image import parse_pages                     # noqa: E402
from pipeline import BACKENDS, Pipeline, process      # noqa: E402
from resource_guard import (Budget, Limits, ResourceShortage,   # noqa: E402
                            read_memory, read_rss_mb, snapshot)

from gate import Busy, JobGate                        # noqa: E402
from upload import UploadError, parse as parse_multipart   # noqa: E402

WORK = ROOT / "work" / "web"
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
THUMB_PAGES = 6                        # 最多给前几页缩略图，别把响应撑爆
THUMB_WIDTH = 380
STARTED_AT = time.time()

logger = logging.getLogger("web")

# 模型参数：前端可以逐项覆盖，键名与 config.yaml 的 tablerec.paddleocr_vl 一致
MODEL_FIELDS = ("base_url", "model", "api_key", "prompt", "max_tokens",
              "temperature", "timeout", "max_side", "repair_attempts")
NUMERIC = {"max_tokens": int, "temperature": float, "timeout": float,
           "max_side": int, "repair_attempts": int}


class Job:
    """一次上传的所有状态，产物落在 work/web/<id>/。"""

    def __init__(self, upload: Path, original: str, cfg, pages,
                 dpi: int | None, preprocess: bool) -> None:
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.dir = WORK / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.upload = self.dir / safe_name(original)
        shutil.move(str(upload), self.upload)
        self.cfg = cfg
        self.pages = pages
        self.dpi = dpi
        self.preprocess = preprocess


@dataclass(frozen=True)
class Settings:
    """服务级限额，全部来自 config.yaml 的 guard 段。"""

    max_upload_mb: int = 60
    max_pages: int = 60
    max_inflight: int = 1
    max_pending: int = 4
    queue_timeout_s: float = 30.0
    max_conns: int = 32
    keep_hours: float = 24
    max_work_mb: int = 4096

    @classmethod
    def from_config(cls, cfg: dict) -> "Settings":
        raw = (cfg or {}).get("guard") or {}
        return cls(max_upload_mb=int(raw.get("max_upload_mb", 60)),
                   max_pages=int(raw.get("max_pages", 60)),
                   max_inflight=int(raw.get("max_inflight", 1)),
                   max_pending=int(raw.get("max_pending", 4)),
                   queue_timeout_s=float(raw.get("queue_timeout_s", 30)),
                   max_conns=int(raw.get("max_conns", 32)),
                   keep_hours=float(raw.get("keep_hours", 24)),
                   max_work_mb=int(raw.get("max_work_mb", 4096)))

    @property
    def max_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


class App:
    """进程级状态：准入控制 + 有界的 Pipeline 缓存。

    Pipeline 装的是 ONNX 会话（几百 MB），绝不能按请求无限增长：
    只保留最近一份，换后端/换模型参数时把旧的丢掉，先释放再加载新的。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gate = JobGate(max_inflight=settings.max_inflight,
                            max_pending=settings.max_pending,
                            wait_timeout=settings.queue_timeout_s)
        self._pipe: Pipeline | None = None
        self._pipe_key: tuple | None = None
        self._lock = threading.Lock()
        self.last_cleanup = 0.0

    @staticmethod
    def _key(cfg) -> tuple:
        model_cfg = cfg["tablerec"].get("paddleocr_vl", {})
        return (cfg["tablerec"]["backend"],
                tuple(sorted((k, str(v)) for k, v in model_cfg.items())))

    def pipeline(self, cfg) -> Pipeline:
        key = self._key(cfg)
        with self._lock:                         # 加载期间只允许一个线程进来
            if self._pipe is not None and self._pipe_key == key:
                return self._pipe
            self._pipe = None                    # 先松手，把旧会话的内存还回去
            self._pipe_key = None
            self._pipe = Pipeline(cfg)
            self._pipe_key = key
            logger.info("模型已加载（backend=%s）", key[0])
            return self._pipe


APP: "App | None" = None


# ---------------------------------------------------------------- 配置与响应
def base_config(form: dict) -> tuple[dict, list[int] | None, int | None]:
    """config.yaml → 请求级配置；顺带解析页范围和 dpi。"""
    cfg = load_config()
    cfg["tablerec"]["backend"] = form.get("backend") or cfg["tablerec"]["backend"]
    if cfg["tablerec"]["backend"] not in BACKENDS:
        raise ValueError(f"backend 只能是 {BACKENDS}")

    model_cfg = cfg["tablerec"]["paddleocr_vl"]
    for name in MODEL_FIELDS:
        if name not in form:                 # 没传这个字段 → 用配置里的
            continue
        value = str(form[name] or "").strip()
        if name in NUMERIC:                  # 数值不能"清空"，空串就沿用默认
            if value:
                model_cfg[name] = NUMERIC[name](value)
            continue
        # 文本传了（哪怕是空串）就是用户的明确意思：清空。别悄悄回落到配置默认值，
        # 否则前端把 model 删空后仍会拿配置里的模型去跑，问题很难看出来。
        model_cfg[name] = value

    pages = parse_pages(form.get("pages") or None)
    dpi = int(form["dpi"]) if form.get("dpi") else None
    return cfg, pages, dpi


def safe_name(name: str) -> str:
    """上传文件名落到磁盘前先清洗：只留字母数字和 .-_ 与中文，去掉路径。"""
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", base).strip("._") or "upload"
    return cleaned[:120]


def pdf_pages(path: Path) -> int:
    import pymupdf

    with pymupdf.open(path) as doc:
        return doc.page_count


def thumb(image_path: Path) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        return ""
    h, w = image.shape[:2]
    scale = THUMB_WIDTH / float(w)
    small = cv2.resize(image, (THUMB_WIDTH, max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def preview_rows(table: dict, limit: int = 8) -> list[list[str]]:
    rows: dict[int, list[dict]] = {}
    for cell in table["cells"]:
        rows.setdefault(cell["row"], []).append(cell)
    out = []
    for r in sorted(rows)[:limit]:
        out.append([(c["ocr"] or "").replace("\n", " ")
                    for c in sorted(rows[r], key=lambda c: c["col"])])
    return out


MAX_SHEET_CELLS = 400_000       # 只有一个 sheet 超过这么多格才截断（≈ 400×1000）


def read_workbook(path: Path) -> dict:
    """把产出的 xlsx 读成前端能直接铺的结构：行值 + 合并区 + 列宽。

    要求"显示全"，所以**不截断行**；只在一个极高的总格数上限处兜底，
    防止极端表格（几千行 × 几十列）把浏览器卡死。触发了会在页面上明确提示。
    """
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path, data_only=True)
    sheets = []
    for ws in wb.worksheets:
        rows, truncated = [], False
        keep = max(1, MAX_SHEET_CELLS // max(1, ws.max_column))
        for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if i > keep:
                truncated = True
                break
            rows.append(["" if value is None else value for value in row])
        width = max((len(row) for row in rows), default=0)
        widths = [ws.column_dimensions[get_column_letter(c)].width or 0
                  for c in range(1, width + 1)]
        sheets.append({
            "name": ws.title, "rows": rows, "widths": widths, "truncated": truncated,
            "merges": [[m.min_row, m.min_col, m.max_row, m.max_col]
                       for m in ws.merged_cells.ranges],
        })
    return {"sheets": sheets}


def summarise(job: Job, doc: dict, elapsed: float) -> dict:
    """把整份 doc 压成前端要的摘要；缩略图只取前几页。"""
    kinds: dict[str, int] = {}
    for page in doc["pages"]:
        for region in page["regions"]:
            if region["type"] == "table":
                kinds[region["kind"]] = kinds.get(region["kind"], 0) + 1
    fallbacks = doc.get("fallbacks", [])
    pages = []
    stem = job.upload.stem
    for i, page in enumerate(doc["pages"]):
        tables = [r for r in page["regions"] if r["type"] == "table"]
        item = {
            "page": page["page"], "width": page["width"], "height": page["height"],
            "table_count": len(tables),
            "char_count": len(page["regions"]) - len(tables),
            "elapsed": page.get("elapsed_s", 0),
            "timings": page.get("timings", {}),
            "tables": [{"kind": t["kind"], "rows": t["n_rows"], "cols": t["n_cols"],
                        "cells": len(t["cells"]),
                        "preview": preview_rows(t)} for t in tables],
        }
        if i < THUMB_PAGES:
            page_png = job.dir / "pages" / f"p{page['page']:03d}.png"
            source = page_png if page_png.exists() else job.upload
            item["thumb"] = (thumb(source) if source.suffix.lower() in ALLOWED
                             and source.suffix.lower() != ".pdf" else "")
        pages.append(item)
    return {"job": job.id, "doc": job.upload.name, "backend": job.cfg["tablerec"]["backend"],
            "elapsed": round(elapsed, 1), "pages": pages,
            "engine": {"requested": doc.get("engine", job.cfg["tablerec"]["backend"]),
                       "kinds": kinds, "fallback_count": len(fallbacks),
                       "fallbacks": fallbacks[:5]},
            "timings": doc.get("timings", {}),
            "outputs": {"json": f"/api/file/{job.id}/{stem}.json",
                        "xlsx": f"/api/file/{job.id}/{stem}.xlsx",
                        "doc": f"/api/doc/{job.id}",
                        "sheets": f"/api/sheets/{job.id}"}}


def probe(base_url: str, api_key: str = "", timeout: float = 4) -> dict:
    """探活：/v1/models 能返回就认为服务在线。"""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        return {"ok": True, "models": models}
    except Exception as exc:                     # noqa: BLE001  探活失败照实报
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- 资源与清理
def cleanup_jobs(settings: Settings, force: bool = False) -> int:
    """按 keep_hours 与 max_work_mb 淘汰旧任务目录，返回删除的目录数。

    不清理的话 work/web 会一直涨，最后把盘写满——那也是"系统失败"的一种。
    """
    if APP is not None and not force:
        if time.time() - APP.last_cleanup < 120:      # 最多两分钟扫一次
            return 0
        APP.last_cleanup = time.time()
    if not WORK.exists():
        return 0
    removed = 0
    now = time.time()
    for item in WORK.iterdir():
        if not item.is_dir():
            if item.name.startswith("upload-") and now - item.stat().st_mtime > 3600:
                item.unlink(missing_ok=True)          # 半路失败留下的临时文件
            continue
        if (now - item.stat().st_mtime) / 3600 > settings.keep_hours:
            shutil.rmtree(item, ignore_errors=True)
            removed += 1

    def size_mb(path: Path) -> float:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)

    left = sorted((d for d in WORK.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    total = sum(size_mb(d) for d in left)
    while total > settings.max_work_mb and left:
        victim = left.pop(0)                          # 最旧的先走
        total -= size_mb(victim)
        shutil.rmtree(victim, ignore_errors=True)
        removed += 1
    if removed:
        logger.info("清理任务产物 %d 个（保留 %d h / 上限 %d MB）",
                    removed, settings.keep_hours, settings.max_work_mb)
    return removed


_MODEL_CACHE: dict = {"at": 0.0, "key": "", "value": {}}


def model_health(cfg, ttl: float = 5.0) -> dict:
    """模型服务在不在、模型名对不对。缓存 `ttl` 秒，避免健康检查被拖慢。"""
    vlm = (cfg.get("tablerec") or {}).get("paddleocr_vl") or {}
    base = str(vlm.get("base_url") or "")
    if not base:
        return {"ok": False, "error": "没配 base_url"}
    key = f"{base}|{vlm.get('api_key')}"
    now = time.time()
    if _MODEL_CACHE["key"] == key and now - _MODEL_CACHE["at"] < ttl:
        return _MODEL_CACHE["value"]
    result = probe(base, str(vlm.get("api_key") or ""), timeout=2)
    result["want"] = vlm.get("model")
    result["matched"] = bool(result.get("ok")) and result["want"] in (result.get("models") or [])
    _MODEL_CACHE.update({"at": now, "key": key, "value": result})
    return result


def health() -> dict:
    """给运维（和你）看的：现在还撑得住吗、谁在跑、排队多长。"""
    cfg = load_config()
    data = snapshot(cfg)
    data["model_service"] = model_health(cfg)
    data.update({"pid": os.getpid(), "uptime_s": round(time.time() - STARTED_AT, 1),
                 "jobs_dir_mb": round(sum(f.stat().st_size for f in WORK.rglob("*")
                                          if f.is_file()) / (1024 * 1024), 1)
                 if WORK.exists() else 0.0})
    if APP is not None:
        st = APP.gate.stats()
        data["gate"] = {"inflight": st.inflight, "pending": st.pending,
                        "max_inflight": st.max_inflight, "max_pending": st.max_pending,
                        "accepted": st.accepted, "rejected": st.rejected,
                        "active_job": st.active_job, "longest_wait_s": st.longest_wait_s}
    return data


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "financial_ocr/1.0"

    # ------------------------------------------------------------ 工具
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 页面和接口都不缓存：之前改完前端、浏览器还在用旧 HTML，症状就是"点了没反应"
        self.send_header("Cache-Control", "no-store, must-revalidate")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, code: int, message: str, extra: dict | None = None) -> None:
        logger.warning("%s %s", code, message)
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8", extra)

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------ GET
    def do_GET(self) -> None:                    # noqa: N802  http.server 约定
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/config":
            cfg = load_config()
            model = dict(cfg["tablerec"]["paddleocr_vl"])
            model.pop("api_key", None)             # 密钥不回给浏览器
            self._json({"backend": cfg["tablerec"]["backend"], "backends": BACKENDS,
                        "model": model, "dpi": cfg["render"]["dpi"],
                        "has_api_key": bool(cfg["tablerec"]["paddleocr_vl"].get("api_key")
                                            not in ("", "EMPTY"))})
        elif path == "/api/health":
            self._json(health())
        elif path == "/api/probe":
            query = dict(pair.split("=", 1) for pair in
                         self.path.split("?", 1)[-1].split("&") if "=" in pair)
            self._json(probe(urllib.parse.unquote(query.get("base_url", "")),
                             urllib.parse.unquote(query.get("api_key", ""))))
        elif path.startswith("/api/summary/"):
            job_dir = self._job_dir(path)
            cached = job_dir / "_summary.json" if job_dir else None
            if not cached or not cached.exists():
                return self._error(404, "这个任务没有摘要（可能是旧版本跑的结果）")
            self._send(200, cached.read_bytes(), "application/json; charset=utf-8")
        elif path.startswith("/api/doc/"):
            self._job_file(path, ".json", "application/json; charset=utf-8")
        elif path.startswith("/api/sheets/"):
            self._job_file(path, ".xlsx", None)
        elif path.startswith("/api/file/"):
            self._download(path)
        else:
            self._error(404, f"没有这个地址：{path}")

    def _job_dir(self, path: str) -> Path | None:
        """从 /api/xxx/<job>[/name] 里取 job 目录，顺手做越权校验。"""
        job_id = path.rstrip("/").split("/")[3]
        if not re.fullmatch(r"[\w.\-]+", job_id):
            return None
        target = (WORK / job_id).resolve()
        if not str(target).startswith(str(WORK.resolve())) or not target.is_dir():
            return None
        return target

    def _job_file(self, path: str, suffix: str, ctype: str | None) -> None:
        """预览接口：xlsx 走 openpyxl 解析成网格，json 原样返回。"""
        job_dir = self._job_dir(path)
        if job_dir is None:
            return self._error(404, "任务不存在或已被删除")
        files = sorted(job_dir.glob(f"*{suffix}"))
        if not files:
            return self._error(404, f"这个任务没有 {suffix} 产物")
        if ctype is None:
            try:
                self._json(read_workbook(files[0]))
            except Exception as exc:             # noqa: BLE001  解析失败要说清
                logger.error("读 Excel 失败 %s：%s", files[0], traceback.format_exc(limit=2))
                self._error(500, f"读 Excel 失败：{type(exc).__name__}: {exc}")
            return
        self._send(200, files[0].read_bytes(), ctype)

    def _download(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) != 5:
            return self._error(400, "下载地址不对")
        job_id, name = parts[3], parts[4]
        if not re.fullmatch(r"[\w.\-]+", job_id) or not re.fullmatch(r"[\w.\-]+", name):
            return self._error(400, "文件名不合法")
        target = (WORK / job_id / name).resolve()
        if not str(target).startswith(str(WORK.resolve())) or not target.exists():
            return self._error(404, "文件不存在或已被删除")
        ctype = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                 if target.suffix == ".xlsx" else "application/json")
        self._send(200, target.read_bytes(), ctype,
                   {"Content-Disposition": f'attachment; filename="{name}"'})

    # ------------------------------------------------------------ POST
    def do_POST(self) -> None:                   # noqa: N802  http.server 约定
        if self.path.split("?")[0] != "/api/run":
            return self._error(404, "没有这个地址")
        settings = APP.settings
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return self._error(411, "请求体是空的")
        if length > settings.max_bytes:
            return self._error(413, f"文件太大（{length // 1024 // 1024}MB，"
                                    f"上限 {settings.max_upload_mb}MB）")

        WORK.mkdir(parents=True, exist_ok=True)
        tmp = WORK / f"upload-{uuid.uuid4().hex[:8]}.part"
        try:
            with open(tmp, "wb") as sink:
                form = parse_multipart(self.rfile, self.headers.get("Content-Type", ""),
                                       sink, max_bytes=settings.max_bytes,
                                       total_length=length)
        except UploadError as exc:
            tmp.unlink(missing_ok=True)
            return self._error(400, f"上传内容不对：{exc}")
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            logger.error("写上传文件失败：%s", exc)
            return self._error(507, f"服务端写盘失败：{exc}")
        except Exception as exc:                 # noqa: BLE001  不能让连接悬着
            tmp.unlink(missing_ok=True)
            logger.error("解析上传失败：%s", traceback.format_exc(limit=3))
            return self._error(400, f"解析上传失败：{type(exc).__name__}: {exc}")

        if form.upload is None:
            tmp.unlink(missing_ok=True)
            return self._error(400, "没有收到文件（表单字段名要是 file）")
        filename = form.upload.filename
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED:
            tmp.unlink(missing_ok=True)
            return self._error(400, f"不支持 {suffix or '这种'} 文件，"
                                    f"只收 {', '.join(sorted(ALLOWED))}")
        try:
            cfg, pages, dpi = base_config(form.fields)
        except ValueError as exc:
            tmp.unlink(missing_ok=True)
            return self._error(400, f"参数不对：{exc}")

        if suffix == ".pdf" and pages is None:
            total = pdf_pages(tmp)
            if total > settings.max_pages:
                tmp.unlink(missing_ok=True)
                return self._error(400, f"这份 PDF 有 {total} 页，一次最多跑 "
                                        f"{settings.max_pages} 页，请在「页范围」里指定（如 114-116）")
        preprocess = form.fields.get("preprocess", "1") not in ("0", "false", "off")
        try:
            self._run_job(tmp, filename, cfg, pages, dpi, preprocess, form.upload.size)
        finally:
            tmp.unlink(missing_ok=True)          # Job 成功后已移走，这里兜底
            cleanup_jobs(settings)

    def _run_job(self, tmp: Path, filename: str, cfg, pages, dpi,
                 preprocess: bool, size: int) -> None:
        """准入 → 预算 → 跑 → 回包。所有资源类失败都转成明确的 429/503。"""
        job: Job | None = None
        done = False
        try:
            with APP.gate.slot(filename) as waited:
                job = Job(tmp, filename, cfg, pages, dpi, preprocess)
                logger.info("任务开始 %s（%.1fMB，backend=%s，页=%s，排队 %.2fs）",
                            job.id, size / 1048576, job.cfg["tablerec"]["backend"],
                            job.pages or "全部", waited)
                budget = Budget(Limits.from_config(cfg), tag=job.id)
                budget.precheck()
                with job_ctx(job.id):
                    t_run = time.perf_counter()
                    doc = process(job.upload, job.dir, job.cfg, pages=job.pages,
                                  dpi=job.dpi, do_preprocess=job.preprocess,
                                  pipe=APP.pipeline(job.cfg), budget=budget)
                    run_s = time.perf_counter() - t_run
                logger.info("任务完成 job=%s 排队=%.2fs 计算=%.2fs 页=%d 峰值RSS=%dMB | %s",
                            job.id, waited, run_s, len(doc["pages"]), read_rss_mb(),
                            " ".join(f"{k}={doc['timings'][k]}ms" for k in TOTAL_ORDER
                                     if k in doc["timings"]))
            summary = summarise(job, doc, doc["elapsed_s"])
            (job.dir / "_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False), encoding="utf-8")
            self._json(summary)
            done = True
        except Busy as exc:
            self._error(429, str(exc), {"Retry-After": "5"})
        except ResourceShortage as exc:
            logger.warning("资源不足，拒绝任务：%s", exc)
            self._error(503, f"资源不足：{exc}", {"Retry-After": "10"})
        except MemoryError:
            logger.error("内存耗尽，已中止这次任务（进程仍在运行）")
            self._error(503, "内存不足，本次任务已中止；请缩小页范围或降低 DPI",
                        {"Retry-After": "10"})
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:                 # noqa: BLE001  前端要看到原因
            logger.error("任务失败：%s", traceback.format_exc(limit=3))
            self._error(500, f"{type(exc).__name__}: {exc}")
        finally:
            # 失败的这一单不要留下半成品目录，否则被拒的任务会一直占着盘
            if not done and job is not None and job.dir.exists():
                shutil.rmtree(job.dir, ignore_errors=True)


class BoundedServer(ThreadingHTTPServer):
    """连接数有上限的 HTTP 服务。

    默认 ThreadingHTTPServer 来一个连一个线程，几百个并发连接就能把线程表和内存吃光。
    这里用信号量兜住：满了直接回 503 并关掉连接，进程不受影响。
    """

    daemon_threads = True          # 客户端半途断开不留僵尸线程
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, addr, handler, max_conns: int) -> None:
        self._conns = threading.BoundedSemaphore(max(1, max_conns))
        self.rejected = 0
        super().__init__(addr, handler)

    def process_request(self, request, client_address) -> None:
        if not self._conns.acquire(blocking=False):
            self.rejected += 1
            logger.warning("连接数超限（%s），直接回 503", client_address[0])
            try:
                body = json.dumps({"error": "服务繁忙：连接数已满，请稍后重试"},
                                  ensure_ascii=False).encode("utf-8")
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\n"
                                b"Content-Type: application/json; charset=utf-8\r\n"
                                b"Retry-After: 5\r\n"
                                + f"Content-Length: {len(body)}\r\n".encode()
                                + b"Connection: close\r\n\r\n" + body)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._conns.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._conns.release()


def main(argv: list[str] | None = None) -> int:
    global APP
    p = argparse.ArgumentParser(description="上传图片/PDF → json + excel 的本地服务")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=Path, default=None)
    args = p.parse_args(argv)
    setup_logging(args.log_level, args.log_file)

    cfg = load_config()
    settings = Settings.from_config(cfg)
    APP = App(settings)
    WORK.mkdir(parents=True, exist_ok=True)
    cleanup_jobs(settings, force=True)

    server = BoundedServer((args.host, args.port), Handler, settings.max_conns)
    mem = read_memory()
    logger.info("服务就绪 http://%s:%d | 并发 %d / 排队 %d / 连接 %d | "
                "内存可用 %dMB | 任务保留 %dh | 日志 %s",
                args.host, args.port, settings.max_inflight, settings.max_pending,
                settings.max_conns, mem.available_mb, int(settings.keep_hours),
                args.log_file or "work/logs/app.log")
    if mem.available_mb < Limits.from_config(cfg).min_free_mb:
        logger.warning("可用内存只有 %dMB，低于门槛 %dMB：现在只会拒任务，不会崩",
                       mem.available_mb, Limits.from_config(cfg).min_free_mb)

    def _stop(signum, _frame) -> None:
        logger.info("收到信号 %s，停止接受新请求（等当前任务收尾）", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        logger.info("服务已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
