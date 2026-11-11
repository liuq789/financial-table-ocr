"""表格识别：整张表交给 PaddleOCR-VL，结构与文字一次出。

为什么单独一条路
----------------
本地链路（有线 unet / 无线读光 + PP-OCR 逐格）的短板是结构：密集财报表会漏
行线（`流动资产：` 和 `货币资金` 挤进同一行带），无线小表会串行。PaddleOCR-VL
直接出一张 HTML 表，行列和合并关系一次到位，不用再猜线。

它跑在本机的 vLLM 上（OpenAI 兼容接口，见 `tools/serve_model.sh`），所以这条
路是"本地模型 + 本地服务"，不是云调用。地址、模型名、提示词在
`config.yaml` 的 `tablerec.paddleocr_vl` 里配。

只连一个 OpenAI 兼容端点、只用标准库（urllib），不为一个 POST 引 openai 包。

模型输出格式
------------
先看有没有 `<td>`/`<th>`（PaddleOCR-VL 的表识别输出），再看有没有 OTSL 标记
（老版本 XLS 格式，兼容用）。都对不上就抛错，不返回空表。

单独跑
------
    # 打本机服务（没起服务会直接报错，不静默返回空表）
    PYTHONPATH=src python -m model.paddleocr_vl --image work/crop.png --out work/table.json

    # 只解析已保存的模型输出，离线调解析规则
    PYTHONPATH=src python -m model.paddleocr_vl --parse-file work/table.html
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.MULTILINE)
_OTSL_TAG = re.compile(r"<(fcel|lcel|ucel|xcel|ecel|nl)>")


class PaddleOcrVlError(RuntimeError):
    """视觉大模型请求失败，或返回的内容不是可解析的表格。"""


class TableTextError(ValueError):
    """模型输出的文本认不出表格格式。"""


@dataclass(frozen=True)
class Cell:
    """一个单元格。bbox 是裁图内的像素坐标 [x0, y0, x1, y1]。"""

    row: int
    col: int
    rowspan: int
    colspan: int
    bbox: tuple[int, int, int, int]
    text: str


@dataclass(frozen=True)
class TableResult:
    """一张表的识别结果。n_rows/n_cols 已包含合并单元格撑开的范围。"""

    n_rows: int
    n_cols: int
    cells: list[Cell]
    raw: str


@dataclass
class _Slot:
    """解析阶段的单元格：span 会被合并标记就地加大，所以不直接用 Cell。"""

    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""

    def freeze(self) -> Cell:
        return Cell(self.row, self.col, self.rowspan, self.colspan, (0, 0, 0, 0), self.text)


# ---------------------------------------------------------------- 文本解析
def _clean(text: str) -> str:
    """去多余空白：行内空格压成一个，保留换行（格内多行要用）。"""
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _strip_fence(text: str) -> str:
    """去掉 ```html ... ``` 这类代码围栏。"""
    return _FENCE.sub("", text or "").strip()


def _span(value: str | None) -> int:
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return 1


class _HtmlTableParser(HTMLParser):
    """只收集 tr/td/th 的文本与 span。不支持单元格里再嵌表格（财报里没有）。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_Slot]] = []
        self._row: list[_Slot] | None = None
        self._cell: _Slot | None = None
        self._buf: list[str] = []

    def _start_row(self) -> None:
        self._row = []
        self.rows.append(self._row)

    def _flush_cell(self) -> None:
        """结束当前单元格。模型经常省掉 </td>，下一个 <td>/<tr> 就是隐式闭合。"""
        if self._cell is None:
            return
        if self._row is None:
            self._start_row()
        self._cell.text = _clean("".join(self._buf))
        self._row.append(self._cell)
        self._cell = None
        self._buf = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attr = {k.lower(): v for k, v in attrs}
        if tag == "tr":
            self._flush_cell()
            self._start_row()
        elif tag in ("td", "th"):
            self._flush_cell()
            if self._row is None:                    # 表格没写 tr，直接来 td
                self._start_row()
            self._cell = _Slot(row=len(self.rows) - 1, col=0,
                               rowspan=_span(attr.get("rowspan")),
                               colspan=_span(attr.get("colspan")))
            self._buf = []
        elif tag == "br" and self._cell is not None:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._flush_cell()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buf.append(data)

    def finish(self) -> None:
        """收尾：全文结束还挂着没闭合的单元格。"""
        self._flush_cell()


def _place(rows: list[list[_Slot]]) -> tuple[list[Cell], int, int]:
    """把按行组织的单元格铺到网格上：被上面 rowspan 占住的位置要跳过。"""
    taken: set[tuple[int, int]] = set()
    cells: list[Cell] = []
    for r, row in enumerate(rows):
        col = 0
        for slot in row:
            while (r, col) in taken:
                col += 1
            for dr in range(slot.rowspan):
                for dc in range(slot.colspan):
                    taken.add((r + dr, col + dc))
            cells.append(replace(slot.freeze(), row=r, col=col))
            col += slot.colspan
    n_rows = max((c.row + c.rowspan for c in cells), default=0)
    n_cols = max((c.col + c.colspan for c in cells), default=0)
    return cells, n_rows, n_cols


def parse_html(text: str) -> tuple[list[Cell], int, int]:
    """`<table>` → 网格。rowspan/colspan 按 HTML 语义展开。"""
    parser = _HtmlTableParser()
    parser.feed(text)
    parser.close()
    parser.finish()
    if not parser.rows:
        raise TableTextError("HTML 里没有 tr/td")
    return _place(parser.rows)


def parse_otsl(text: str) -> tuple[list[Cell], int, int]:
    """OTSL 标记串 → 网格。

    fcel 有内容 / ecel 空 / lcel 与左合并 / ucel 与上合并 / xcel 左+上都合并。
    合并统一处理：找到目标格，缺哪一维就补哪一维；已经覆盖到当前位置就跳过
    （2×2 整块合并时 xcel 会落在一个本来就盖住的格子上，重复加会撑成 3×3）。
    """
    first = _OTSL_TAG.search(text)
    if first is None:
        raise TableTextError("没有 OTSL 标记")
    parts = _OTSL_TAG.split(text[first.start():])
    slots: list[_Slot] = []
    owner: dict[tuple[int, int], _Slot] = {}
    row = col = 0

    for i in range(1, len(parts), 2):
        tag = parts[i]
        buf = _clean(parts[i + 1]) if i + 1 < len(parts) else ""
        if tag == "nl":
            row, col = row + 1, 0
            continue
        while (row, col) in owner:                   # 被上面的 rowspan 占住
            col += 1

        if tag in ("lcel", "ucel", "xcel"):
            left, up = owner.get((row, col - 1)), owner.get((row - 1, col))
            target = left if tag == "lcel" else (up or left)
            if target is not None:
                if target.col + target.colspan <= col:
                    target.colspan += 1
                if target.row + target.rowspan <= row:
                    target.rowspan += 1
                owner[(row, col)] = target
            else:                                    # 合并目标缺失：降级成普通格
                slot = _Slot(row=row, col=col, text=buf)
                slots.append(slot)
                owner[(row, col)] = slot
        else:                                        # fcel / ecel
            slot = _Slot(row=row, col=col, text=buf)
            slots.append(slot)
            owner[(row, col)] = slot
        col += 1

    cells = [s.freeze() for s in slots]
    n_rows = max((c.row + c.rowspan for c in cells), default=0)
    n_cols = max((c.col + c.colspan for c in cells), default=0)
    return cells, n_rows, n_cols


def parse_table_text(text: str) -> tuple[list[Cell], int, int]:
    """HTML 优先，其次 OTSL；都不是就抛错，不猜。"""
    body = _strip_fence(text)
    if not body:
        raise TableTextError("模型返回空内容")
    low = body.lower()
    if "<td" in low or "<th" in low:
        return parse_html(body)
    if _OTSL_TAG.search(body):
        return parse_otsl(body)
    raise TableTextError(f"认不出表格格式：{body[:120]!r}")


def grid_problem(cells: list[Cell], n_rows: int, n_cols: int) -> str | None:
    """检查网格是否可信，返回问题描述；没问题返回 None。

    这三条是"模型看着像给了张表、其实列丢了"的典型症状：行列号冲突（同一格被
    两个单元格覆盖）、空洞（某几列整列没输出）、大量空格（数值列整体丢失）。
    """
    if n_rows < 2 or n_cols < 2:
        return f"网格太小 {n_rows}×{n_cols}"
    cover = [[0] * n_cols for _ in range(n_rows)]
    for cell in cells:
        for dr in range(cell.rowspan):
            for dc in range(cell.colspan):
                r, c = cell.row + dr, cell.col + dc
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    cover[r][c] += 1
    dup = sum(1 for row in cover for v in row if v > 1)
    if dup:
        return f"{dup} 个位置被重复覆盖（行列号冲突）"
    holes = sum(1 for row in cover for v in row if v == 0)
    if holes:
        return f"{holes} 个空洞（行列号不连续）"
    empty = sum(1 for cell in cells if not cell.text.strip())
    if len(cells) >= 20 and empty / len(cells) > 0.6:
        return f"{empty}/{len(cells)} 个格子为空，疑似整列丢失"
    return None


def assign_bboxes(cells: list[Cell], width: int, height: int) -> list[Cell]:
    """按行列均分出像素框。

    大模型只给行列号，不给坐标。均分框不是真实位置，只用于排序、预览和
    excel 的列宽估计——**别拿它当单元格定位结果用**。
    """
    n_rows = max((c.row + c.rowspan for c in cells), default=0)
    n_cols = max((c.col + c.colspan for c in cells), default=0)
    if n_rows <= 0 or n_cols <= 0:
        return cells
    ys = [round(height * r / n_rows) for r in range(n_rows + 1)]
    xs = [round(width * c / n_cols) for c in range(n_cols + 1)]
    out = []
    for c in cells:
        r1 = min(n_rows, c.row + c.rowspan)
        c1 = min(n_cols, c.col + c.colspan)
        out.append(replace(c, bbox=(xs[c.col], ys[c.row], xs[c1], ys[r1])))
    return out


# ---------------------------------------------------------------- 请求
PATCH = 28          # 视觉编码的等效 patch 边长，用来估算"这张图要多少 token"
MIN_SIDE = 320      # 缩到这个尺寸还不行就认输，让上层兜底（不能无限缩）
OVERFLOW_HINTS = ("context length", "maximum context", "too long", "too many tokens",
                  "exceeds", "超出", "超过最大")


def _estimate_tokens(height: int, width: int, prompt_len: int = 0) -> int:
    """估算输入 token：按 patch 数 + 提示词 + 余量。宁可高估，不要低估。"""
    patches = math.ceil(height / PATCH) * math.ceil(width / PATCH)
    return patches + prompt_len + 32


def _fit_image(image: np.ndarray, max_side: int, max_pixels: int
               ) -> tuple[np.ndarray, float]:
    """两道上限：长边 ≤ max_side，总像素 ≤ max_pixels。返回 (图, 缩放比)。"""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w))) if max(h, w) > max_side else 1.0
    if max_pixels > 0 and h * w * scale * scale > max_pixels:
        scale *= math.sqrt(max_pixels / (h * w * scale * scale))
    if scale >= 0.999:
        return image, 1.0
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def _encode(image: np.ndarray) -> str:
    """BGR → PNG → data URL（尺寸在上游已经按上限缩过）。"""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise PaddleOcrVlError("裁图编码 PNG 失败")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _is_overflow(message: str) -> bool:
    """服务端是不是在抱怨"输入太长"（这种事应当缩小重发，而不是直接失败）。"""
    low = message.lower()
    return any(hint in low for hint in OVERFLOW_HINTS)


class PaddleOcrVl:
    """视觉大模型表格识别（OpenAI 兼容接口，参数全来自配置）。

    除了"调用 + 解析"，这里还负责三件让模型侧不会失败的事：
    1. 把图缩到上下文/显存撑得住的尺寸（`max_pixels`），并按 `/v1/models` 报的
       `max_model_len` 反推 `max_tokens`，避免把服务端顶成 400。
    2. 服务端真报"上下文超长"时，自动缩小重发（`shrink_attempts`）。
    3. 单表总时间预算（`total_timeout`）：重试 + 重问加起来超过就收手，不让一个
       卡住的请求把整个队列拖死。
    """

    def __init__(self, cfg) -> None:
        self.base_url = str(cfg["base_url"]).rstrip("/")
        self.model = str(cfg["model"])
        self.api_key = str(cfg["api_key"])
        self.prompt = str(cfg["prompt"])
        self.max_tokens = int(cfg["max_tokens"])
        self.temperature = float(cfg["temperature"])
        self.timeout = float(cfg["timeout"])
        self.retries = int(cfg["retries"])
        self.max_side = int(cfg["max_side"])
        self.max_pixels = int(cfg.get("max_pixels", 0) or 0)
        self.total_timeout = float(cfg.get("total_timeout", 300) or 300)
        self.shrink_attempts = int(cfg.get("shrink_attempts", 2) or 0)
        self.repair_attempts = int(cfg.get("repair_attempts", 1))
        self._limit: int | None = None       # 服务端 max_model_len，查到就缓存
        self._limit_at = 0.0
        if not self.model:
            raise ValueError("没有配置模型名：请填 config.yaml 里 tablerec.paddleocr_vl.model"
                             "（服务端 /v1/models 返回的 id）")
        self.repair_prompt = str(cfg.get("repair_prompt") or
                                 "只输出 <table>…</table>，不要解释、不要代码围栏；"
                                 "每个 <td> 都要写结束标签，行列数与图片完全一致。")

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def model_limit(self) -> int | None:
        """问一次服务端 `/v1/models` 拿 max_model_len（缓存 60s）。

        拿不到就返回 None：宁可不知道，也不能因此让整次识别失败。
        """
        now = time.time()
        if self._limit is not None and now - self._limit_at < 60:
            return self._limit
        url = f"{self.base_url}/models"
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=min(self.timeout, 5)) as resp:
                data = json.loads(resp.read().decode("utf-8") or "{}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            logger.debug("查 max_model_len 失败（不影响识别）：%s", exc)
            self._limit_at = now
            return None
        limit = None
        for item in data.get("data", []):
            if isinstance(item, dict) and item.get("id") == self.model:
                limit = item.get("max_model_len")
                break
        self._limit = int(limit) if limit else None
        self._limit_at = now
        return self._limit

    def _budget(self, est_in: int, limit: int | None) -> int:
        """给输出留多少 token：既不超过配置，也不能把上下文顶爆。"""
        if not limit:
            return self.max_tokens
        room = limit - est_in - 16
        if room < 256:
            return 0                          # 0 表示"这张图必须再缩"
        return max(256, min(self.max_tokens, room))

    def _payload(self, data_url: str, prompt: str, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]}],
        }

    def _shrink(self, image: np.ndarray, factor: float) -> np.ndarray | None:
        """再缩一圈；已经太小就不缩了（返回 None 表示放弃）。"""
        h, w = image.shape[:2]
        if min(h, w) <= MIN_SIDE:
            return None
        size = (max(MIN_SIDE, int(w * factor)), max(MIN_SIDE, int(h * factor)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def predict(self, image: np.ndarray) -> TableResult:
        """表格裁图（BGR）→ 单元格网格。

        流程：按上限缩图 → 查上下文并算 max_tokens → 请求 → 解析 → 网格校验。
        输出不合格换提示词重问；服务端抱怨太长就缩小重发；时间预算超了就收手。
        最后仍不成功才抛 PaddleOcrVlError，由上层退回本地链路（不会让整页失败）。
        """
        height, width = image.shape[:2]
        deadline = time.perf_counter() + self.total_timeout
        work, scale = _fit_image(image, self.max_side, self.max_pixels)
        limit = self.model_limit()
        budget = self._budget(_estimate_tokens(work.shape[0], work.shape[1],
                                               len(self.prompt)), limit)
        while budget == 0:                    # 图本身就把上下文吃满了 → 继续缩
            smaller = self._shrink(work, 0.7)
            if smaller is None:
                raise PaddleOcrVlError(
                    f"裁图缩到最小仍超出模型上下文（{limit}），请缩小页范围或降低 DPI")
            work = smaller
            budget = self._budget(_estimate_tokens(work.shape[0], work.shape[1],
                                                   len(self.prompt)), limit)
        data_url = _encode(work)
        logger.info("模型请求 model=%s 原图=%dx%d 送图=%dx%d(缩%.2f, base64 %.0fKB) "
                    "估算in≈%d max_tokens=%d 上下文=%s",
                    self.model, width, height, work.shape[1], work.shape[0], scale,
                    len(data_url) / 1024.0,
                    _estimate_tokens(work.shape[0], work.shape[1], len(self.prompt)),
                    budget, limit or "未知")

        problem = "模型没有返回可用表格"
        attempt = 0
        shrinks = 0
        while True:
            if time.perf_counter() > deadline:
                raise PaddleOcrVlError(
                    f"超过单表时间预算 {self.total_timeout:.0f}s（已缩图 {shrinks} 次）")
            prompt = self.prompt if attempt == 0 else self.repair_prompt
            t0 = time.perf_counter()
            try:
                text, usage = self._chat(self._payload(data_url, prompt, budget), deadline)
            except PaddleOcrVlError as exc:
                if shrinks < self.shrink_attempts and _is_overflow(str(exc)):
                    smaller = self._shrink(work, 0.7)
                    if smaller is not None:
                        shrinks += 1
                        work = smaller
                        budget = max(256, self._budget(
                            _estimate_tokens(work.shape[0], work.shape[1],
                                             len(self.prompt)), limit))
                        data_url = _encode(work)
                        attempt = 0           # 缩图后用原提示词重来
                        logger.warning("服务端说太长，缩到 %dx%d 重发（第 %d 次）：%s",
                                       work.shape[1], work.shape[0], shrinks,
                                       str(exc)[:120])
                        continue
                raise
            elapsed = time.perf_counter() - t0

            done = usage.get("completion_tokens")
            speed = (f" {done / elapsed:.1f}tok/s" if done and elapsed > 0 else "")
            logger.info("模型返回 %.2fs 第%d次 输出=%d字 用量=%s%s",
                        elapsed, attempt + 1, len(text),
                        f"in={usage.get('prompt_tokens')}/out={done}"
                        if done else usage or "未返回", speed)
            try:
                cells, n_rows, n_cols = parse_table_text(text)
            except TableTextError as exc:
                problem = f"输出解析失败：{exc}"
            else:
                parse_problem = grid_problem(cells, n_rows, n_cols)
                if parse_problem is None:
                    logger.info("模型网格 %d×%d %d格 合并=%d 用时 %.2fs",
                                n_rows, n_cols, len(cells),
                                sum(1 for c in cells if c.rowspan > 1 or c.colspan > 1),
                                elapsed)
                    return TableResult(n_rows, n_cols,
                                       assign_bboxes(cells, width, height), text)
                problem = parse_problem
            logger.warning("表格输出不合格（第 %d/%d 次，%.2fs）：%s",
                           attempt + 1, self.repair_attempts + 1, elapsed, problem)
            attempt += 1
            if attempt > self.repair_attempts:
                raise PaddleOcrVlError(problem)

    def _chat(self, payload: dict, deadline: float | None = None) -> tuple[str, dict]:
        """发一次请求，返回 (文本, usage)。

        重试只针对 408/409/429/5xx/连不上；连接被拒（服务没起）立刻放弃，
        因为重试只会白等。单次超时取 `timeout` 与剩余预算的较小值。
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            wait = self.timeout
            if deadline is not None:
                left = deadline - time.perf_counter()
                if left <= 1:
                    raise PaddleOcrVlError("单表时间预算用完，放弃这次模型调用")
                wait = min(wait, left)
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=wait) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = data.get("usage") or {}
                logger.debug("HTTP %s %.2fs usage=%s", self.endpoint,
                             time.perf_counter() - t0, usage or "无")
                return self._content(data), usage
            except urllib.error.HTTPError as exc:     # 4xx 除了限流都不该重试
                detail = exc.read().decode("utf-8", "replace")[:400]
                logger.warning("HTTP %d %.2fs %s", exc.code,
                               time.perf_counter() - t0, detail[:200])
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise PaddleOcrVlError(
                        f"{self.endpoint} 返回 {exc.code}：{detail}") from exc
                last = PaddleOcrVlError(f"{self.endpoint} 返回 {exc.code}：{detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, ConnectionRefusedError):
                    # 连接被拒 = 服务没起来，重试只会白等，直接报出去让上层兜底
                    raise PaddleOcrVlError(
                        f"{self.endpoint} 连接被拒绝：模型服务没起来？"
                        f"先跑 tools/serve_model.sh") from exc
                logger.warning("请求异常 %.2fs %s", time.perf_counter() - t0, exc)
                last = PaddleOcrVlError(f"连不上 {self.endpoint}：{exc}")
            if attempt < self.retries:
                delay = min(2.0 ** attempt, 8.0)
                if deadline is not None:
                    delay = max(0.2, min(delay, deadline - time.perf_counter() - 1))
                logger.warning("模型请求失败（%s），%.1fs 后重试", last, delay)
                time.sleep(delay)
        raise PaddleOcrVlError(str(last) if self.retries == 0
                               else f"重试 {self.retries} 次仍失败：{last}") from last

    @staticmethod
    def _content(payload: dict) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PaddleOcrVlError(f"响应结构不对：{str(payload)[:300]}") from exc
        if isinstance(content, list):                 # 少数服务返回分块内容
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
        if not str(content).strip():
            raise PaddleOcrVlError("模型返回空字符串")
        return str(content)


# ---------------------------------------------------------------- 单独跑
def _dump(result: TableResult) -> dict:
    return {"engine": "paddleocr_vl", "n_rows": result.n_rows, "n_cols": result.n_cols,
            "cells": [{"row": c.row, "col": c.col, "rowspan": c.rowspan,
                       "colspan": c.colspan, "bbox": list(c.bbox), "ocr": c.text}
                      for c in result.cells]}


def main(argv: list[str] | None = None) -> int:
    from config import load_config

    p = argparse.ArgumentParser(description="表格识别（PaddleOCR-VL）")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="表格裁图")
    src.add_argument("--parse-file", type=Path, help="只解析已保存的模型输出")
    p.add_argument("--out", type=Path, default=None, help="结果 json 路径")
    p.add_argument("--crop", help="从图里裁一块 x0,y0,x1,y1")
    p.add_argument("--url", default=None, help="覆盖 base_url")
    p.add_argument("--model", default=None, help="覆盖模型名")
    p.add_argument("--prompt", default=None, help="覆盖提示词")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.parse_file:
        text = args.parse_file.read_text(encoding="utf-8")
        cells, n_rows, n_cols = parse_table_text(text)
        print(f"{n_rows} 行 × {n_cols} 列，{len(cells)} 格")
        for c in cells:
            span = (f"r{c.rowspan}" if c.rowspan > 1 else "") + \
                   (f"c{c.colspan}" if c.colspan > 1 else "")
            print(f"  r{c.row:02d} c{c.col:02d} {span:6s} {c.text[:40]!r}")
        return 0

    cfg = load_config(args.config)["tablerec"]["paddleocr_vl"]
    cfg = dict(cfg)
    for key, value in (("base_url", args.url), ("model", args.model),
                       ("prompt", args.prompt)):
        if value:
            cfg[key] = value

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"读不了图片：{args.image}")
        return 1
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
        image = image[y0:y1, x0:x1]

    t0 = time.time()
    result = PaddleOcrVl(cfg).predict(image)
    print(f"{result.n_rows} 行 × {result.n_cols} 列，{len(result.cells)} 格，"
          f"{time.time() - t0:.1f}s")

    out = args.out or (ROOT / "work" / f"table_{args.image.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_dump(result), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
