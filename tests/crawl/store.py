"""落盘 + 台账。所有"文件放哪、下过什么"的规则只在这里。

目录（相对 config.yaml 的 data）：

    data/raw/<代码>/<年>_<类型>.pdf          原件（有预算才留）
    data/pages/<代码>/<年>_<类型>/pNNN.png   表格页图，200 DPI
    data/pages/<代码>/<年>_<类型>/index.json 该份报告的页清单
    data/crawl.sqlite                        台账：去重、断点续爬、配额统计

台账三态：new → done / failed。failed 带 error 文本，重跑时按 attempts 上限重试。

单独跑：
    PYTHONPATH=src:tests python -m crawl.store --stats
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from crawl.contracts import Announcement, CrawlSettings, DocOutcome, RenderedPage

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    key         TEXT PRIMARY KEY,
    sec_code    TEXT NOT NULL,
    sec_name    TEXT,
    title       TEXT,
    year        TEXT,
    kind        TEXT,
    url         TEXT NOT NULL,
    size_kb     INTEGER DEFAULT 0,
    published   TEXT,
    status      TEXT NOT NULL DEFAULT 'new',
    attempts    INTEGER NOT NULL DEFAULT 0,
    n_pages     INTEGER DEFAULT 0,
    n_images    INTEGER DEFAULT 0,
    bytes_pdf   INTEGER DEFAULT 0,
    bytes_img   INTEGER DEFAULT 0,
    kept_pdf    INTEGER DEFAULT 0,
    error       TEXT,
    updated     REAL
);
CREATE INDEX IF NOT EXISTS idx_status ON docs(status);
"""


class Store:
    """线程安全（所有 DB 操作过一把锁）。文件操作本身幂等，不需要锁。"""

    def __init__(self, cfg: CrawlSettings) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None

    # ---------------------------------------------------------------- 生命周期
    def open(self) -> Store:
        self.cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.pages_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.cfg.db_path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")     # 允许"一边抓一边登记目录"
        self._db.executescript(_SCHEMA)
        self._db.commit()
        return self

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> Store:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- 台账
    def add(self, ann: Announcement) -> bool:
        """登记待办。已存在则只补全元数据，返回 False。"""
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO docs(key,sec_code,sec_name,title,year,kind,url,size_kb,"
                "published,status,updated) VALUES(?,?,?,?,?,?,?,?,?,'new',?)",
                (ann.key, ann.sec_code, ann.sec_name, ann.title, ann.year, ann.kind,
                 ann.url, ann.size_kb, ann.published, time.time()),
            )
            self._db.commit()
            return cur.rowcount > 0

    def pending(self, limit: int, max_attempts: int = 3) -> list[Announcement]:
        """取待处理的：没跑过的优先，其次是失败但没到次数上限的。"""
        sql = (
            "SELECT * FROM docs WHERE status='new' OR (status='failed' AND attempts<?) "
            "ORDER BY CASE status WHEN 'new' THEN 0 ELSE 1 END, rowid LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(sql, (max_attempts, limit)).fetchall()
        return [_to_ann(r) for r in rows]

    def mark_running(self, key: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE docs SET attempts=attempts+1, updated=? WHERE key=?", (time.time(), key)
            )
            self._db.commit()

    def mark_done(self, out: DocOutcome) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE docs SET status='done', n_pages=?, n_images=?, bytes_pdf=?, bytes_img=?,"
                " kept_pdf=?, error=NULL, updated=? WHERE key=?",
                (out.n_pages, out.n_images, out.pdf_bytes if out.kept_pdf else 0,
                 out.image_bytes, int(out.kept_pdf), time.time(), out.ann.key),
            )
            self._db.commit()

    def mark_failed(self, key: str, error: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE docs SET status='failed', error=?, updated=? WHERE key=?",
                (error[:500], time.time(), key),
            )
            self._db.commit()

    def stats(self) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT status, COUNT(*) n, SUM(n_images) img, SUM(bytes_img) ib, "
                "SUM(CASE WHEN kept_pdf=1 THEN bytes_pdf ELSE 0 END) pb FROM docs GROUP BY status"
            ).fetchall()
            total = self._db.execute("SELECT COUNT(*), SUM(n_images) FROM docs").fetchone()
        out = {"total": total[0] or 0, "images": total[1] or 0,
               "bytes_img": 0, "bytes_pdf": 0, "by_status": {}}
        for r in row:
            out["by_status"][r["status"]] = {"docs": r["n"], "images": r["img"] or 0}
            out["bytes_img"] += r["ib"] or 0
            out["bytes_pdf"] += r["pb"] or 0
        return out

    # ---------------------------------------------------------------- 文件
    def save_pdf(self, ann: Announcement, data: bytes) -> Path:
        path = self.cfg.raw_dir / ann.sec_code / f"{ann.slug}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, data)
        return path

    def drop_pdf(self, ann: Announcement) -> None:
        path = self.cfg.raw_dir / ann.sec_code / f"{ann.slug}.pdf"
        path.unlink(missing_ok=True)

    def page_dir(self, ann: Announcement) -> Path:
        return self.cfg.pages_dir / ann.sec_code / ann.slug

    def save_pages(self, ann: Announcement, n_pages: int, pages: list[RenderedPage]) -> int:
        """落页图 + index.json。返回真正写下去的页数。"""
        if not pages:
            return 0
        out_dir = self.page_dir(ann)
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for page in pages:
            name = f"p{page.hit.page_no:03d}.png"
            _atomic_write(out_dir / name, page.png)
            entries.append(
                {
                    "page": page.hit.page_no,
                    "file": name,
                    "rows": page.hit.rows,
                    "cols": page.hit.n_cols,
                    "chars": page.hit.chars,
                    "boxes": [list(b) for b in page.hit.boxes],
                    "width": page.width,
                    "height": page.height,
                    "dpi": page.dpi,
                }
            )
        index = {
            "sec_code": ann.sec_code,
            "sec_name": ann.sec_name,
            "year": ann.year,
            "kind": ann.kind,
            "title": ann.title,
            "url": ann.url,
            "published": ann.published,
            "pages_total": n_pages,
            "pages_kept": len(entries),
            "pages": entries,
        }
        _atomic_write(
            out_dir / "index.json",
            json.dumps(index, ensure_ascii=False, indent=1).encode("utf-8"),
        )
        return len(entries)

    def free_gb(self) -> float:
        return shutil.disk_usage(self.cfg.data_root).free / 2**30


def _to_ann(row: sqlite3.Row) -> Announcement:
    return Announcement(
        sec_code=row["sec_code"], sec_name=row["sec_name"] or "", title=row["title"] or "",
        year=row["year"] or "", kind=row["kind"] or "annual", url=row["url"],
        size_kb=row["size_kb"] or 0, published=row["published"] or "",
    )


def _atomic_write(path: Path, data: bytes) -> None:
    """先写 .part 再改名，进程被杀也不会留下半张图。"""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _main() -> int:
    import argparse

    from config import load_config, resolve
    from crawl.settings import build_settings

    ap = argparse.ArgumentParser(description="台账统计")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with Store(build_settings(load_config(), resolve)) as store:
        st = store.stats()
    print(json.dumps(st, ensure_ascii=False, indent=2))
    print(f"磁盘剩余 {shutil.disk_usage(Path.cwd()).free / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
