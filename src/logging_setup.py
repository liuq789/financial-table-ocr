"""日志与计时：全项目只在这里配置一次。

为什么单独一个模块：`logging.basicConfig` 散在各入口会互相覆盖（web 起完 CLI 再起就
换格式了），而且"慢在哪"这个问题必须靠**阶段计时**回答，不能靠临时 print。

用法：
    from logging_setup import setup_logging, Stopwatch, job_ctx

    setup_logging()                      # 幂等，重复调用不会加第二个 handler
    with job_ctx("20261110-1200-ab12"):  # 日志行会带 [job=...]
        sw = Stopwatch()
        with sw.span("layout"): ...
        logger.info("完成 %s", sw.line())

产物：
    控制台（stderr）：`20:07:31.412 INFO  pipeline: ...`（不开日期，屏幕窄）
    文件：`work/logs/app.log`（带日期、job、模块，10MB × 3 滚动）
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "work" / "logs"

_JOB = contextvars.ContextVar("job", default="-")


class _JobFilter(logging.Filter):
    """把当前 job id 塞进 record，格式串里就能用 %(job)s。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.job = _JOB.get()
        return True


@contextlib.contextmanager
def job_ctx(name: str):
    """给一段代码打上任务标记，异常也会还原。"""
    token = _JOB.set(name)
    try:
        yield name
    finally:
        _JOB.reset(token)


def setup_logging(level: int | str = logging.INFO, log_file: Path | None = None,
                  quiet: bool = False) -> None:
    """配置根 logger。幂等：重复调用只调等级，不再加 handler。"""
    root = logging.getLogger()
    root.setLevel(level)
    if getattr(root, "_focr_ready", False):
        return

    fmt_console = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S")
    fmt_file = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s [job=%(job)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt_console)
        console.addFilter(_JobFilter())
        root.addHandler(console)

    path = Path(log_file) if log_file else LOG_DIR / "app.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024,
                                       backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt_file)
    file_handler.addFilter(_JobFilter())
    root.addHandler(file_handler)

    # 这些库在 INFO 下太吵，压到 WARNING
    for noisy in ("onnxruntime", "PIL", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._focr_ready = True                       # type: ignore[attr-defined]
    root.debug("日志已就绪：%s", path)


class _Span:
    """`with sw.span("name")` 用。同名 span 会自动累加。"""

    def __init__(self, watch: "Stopwatch", name: str) -> None:
        self.watch = watch
        self.name = name

    def __enter__(self) -> "_Span":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.watch.add(self.name, (time.perf_counter() - self.t0) * 1000.0)


class Stopwatch:
    """累加命名阶段耗时（毫秒），最后一次性写成日志/JSON。

    计时口径统一用 perf_counter（单调时钟），跟"现在几点"无关。
    """

    def __init__(self) -> None:
        self._ms: dict[str, float] = {}

    def add(self, name: str, ms: float) -> None:
        self._ms[name] = self._ms.get(name, 0.0) + ms

    def span(self, name: str) -> _Span:
        return _Span(self, name)

    def as_dict(self) -> dict[str, int]:
        return {k: int(round(v)) for k, v in self._ms.items()}

    def line(self, order: tuple[str, ...] | None = None) -> str:
        """`detect=812ms layout=95ms ...`，按固定顺序输出，方便肉眼对齐和 grep。"""
        keys = order or tuple(self._ms)
        parts = [f"{k}={int(round(self._ms[k]))}ms" for k in keys if k in self._ms]
        return " ".join(parts)

    def total_ms(self) -> int:
        return int(round(sum(self._ms.values())))


TOTAL_ORDER = ("render", "detect", "preprocess", "layout", "table", "model",
               "ocr", "post", "export")
