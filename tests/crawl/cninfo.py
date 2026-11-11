"""巨潮资讯网（cninfo）公告检索与 PDF 下载。全项目只有这里有网络。

接口事实（2026-11 实测）：
    检索  POST http://www.cninfo.com.cn/new/hisAnnouncement/query
          表单：pageNum / pageSize / column=szse / tabName=fulltext /
                category=category_ndbg_szsh / seDate=起~止
          返回：{"totalRecordNum": N, "announcements": [...]}
    下载  http://static.cninfo.com.cn/ + adjunctUrl

踩过的坑：
    * 必须带 UA 和 Referer，否则返回 HTML 而不是 JSON；
    * `category` 单独用会漏（同一家公司有多个 columnId），所以还要按标题正则再筛；
    * `pageSize` 上限 30，填 50 会报错；
    * `adjunctSize` 单位是 KB。

单独跑：
    PYTHONPATH=src:tests python -m crawl.cninfo --year 2023 --limit 5
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from crawl.contracts import Announcement, CrawlSettings

log = logging.getLogger(__name__)

QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_URL = "http://static.cninfo.com.cn/"
REFERER = "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

CATEGORY = {"annual": "category_ndbg_szsh", "semiannual": "category_bndbg_szsh"}

# 只要正报全文：摘要/英文版/更正/问询回复/审计报告这些都不是我们要的表格素材
_REJECT = re.compile(
    r"摘要|英文|English|已取消|更正|补充|问询|回复|意见|审计报告|内部控制|"
    r"社会责任|ESG|募集资金|提示性公告|说明|核查|专项|独立董事|监事会决议"
)
_YEAR = {
    "annual": re.compile(r"(20\d{2})\s*年年度报告"),
    "semiannual": re.compile(r"(20\d{2})\s*年半年度报告"),
}


class FetchError(RuntimeError):
    """下载/检索失败。permanent=True 表示重试也没用（404/不是 PDF）。"""

    def __init__(self, msg: str, permanent: bool = False) -> None:
        super().__init__(msg)
        self.permanent = permanent


def _request(url: str, data: bytes | None = None) -> bytes:
    req = urllib.request.Request(url, data=data)
    req.add_header("User-Agent", UA)
    req.add_header("Referer", REFERER)
    req.add_header("Accept", "*/*")
    if data is not None:
        req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body


def is_report(title: str, kind: str) -> bool:
    """标题是不是我们要的正报全文（可单独调用，便于测试）。"""
    if _REJECT.search(title):
        return False
    return bool(_YEAR[kind].search(title))


def to_announcement(raw: dict, kind: str) -> Announcement | None:
    """接口原始条目 → 契约。不合格返回 None。"""
    title = (raw.get("announcementTitle") or "").replace("<em>", "").replace("</em>", "")
    if not is_report(title, kind):
        return None
    url = raw.get("adjunctUrl") or ""
    if not url.lower().endswith(".pdf"):
        return None
    m = _YEAR[kind].search(title)
    ts = raw.get("announcementTime") or 0
    return Announcement(
        sec_code=str(raw.get("secCode") or ""),
        sec_name=(raw.get("secName") or "").strip(),
        title=title.strip(),
        year=m.group(1) if m else "",
        kind=kind,
        url=STATIC_URL + url.lstrip("/"),
        size_kb=int(raw.get("adjunctSize") or 0),
        published=time.strftime("%Y-%m-%d", time.gmtime(ts / 1000)) if ts else "",
    )


class CninfoClient:
    """检索 + 下载。构造时传入运行时参数，避免读全局配置。"""

    def __init__(self, settings: CrawlSettings) -> None:
        self.cfg = settings

    def iter_announcements(
        self, kind: str, year: str, limit: int = 0, max_pages: int = 400
    ) -> Iterator[Announcement]:
        """按"披露年份"翻页。year 是公告发布年，不是报告年度。

        同一 key 只出一次（修订稿、二次披露会被折叠）。
        """
        seen: set[str] = set()
        emitted = 0
        for page in range(1, max_pages + 1):
            raws = self._query_page(kind, year, page)
            if not raws:
                return
            for raw in raws:
                ann = to_announcement(raw, kind)
                if ann is None or ann.key in seen:
                    continue
                seen.add(ann.key)
                yield ann
                emitted += 1
                if limit and emitted >= limit:
                    return
            time.sleep(self.cfg.min_delay_s)

    def _query_page(self, kind: str, year: str, page: int) -> list[dict]:
        form = {
            "pageNum": str(page),
            "pageSize": "30",
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": "",
            "secid": "",
            "category": CATEGORY[kind],
            "trade": "",
            "seDate": f"{year}-01-01~{year}-12-31",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        body = urllib.parse.urlencode(form).encode()
        payload = self._retry(lambda: _request(QUERY_URL, body), f"检索 {kind} {year} p{page}")
        try:
            data = json.loads(payload.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise FetchError(f"检索返回不是 JSON：{payload[:120]!r}") from exc
        if page == 1:
            log.info("%s %s 共 %s 条公告", kind, year, data.get("totalRecordNum"))
        return data.get("announcements") or []

    def download(self, ann: Announcement) -> bytes:
        """下 PDF。返回字节；失败抛 FetchError（区分可重试/永久）。"""
        data = self._retry(lambda: _request(ann.url), f"下载 {ann.slug}")
        if not data.startswith(b"%PDF"):
            raise FetchError(f"{ann.slug} 不是 PDF（{len(data)} B，头 {data[:40]!r}）", permanent=True)
        return data

    def _retry(self, fn, what: str) -> bytes:
        last: Exception | None = None
        for attempt in range(1, self.cfg.retries + 1):
            try:
                return fn()
            except urllib.error.HTTPError as exc:
                last = exc
                permanent = exc.code in (400, 403, 404, 410)
                log.warning("%s 第 %d 次失败：HTTP %s", what, attempt, exc.code)
                if permanent:
                    raise FetchError(f"{what}: HTTP {exc.code}", permanent=True) from exc
            except (urllib.error.URLError, OSError) as exc:
                last = exc
                log.warning("%s 第 %d 次失败：%s", what, attempt, exc)
            if attempt < self.cfg.retries:
                time.sleep(self.cfg.retry_wait_s * attempt + random.random() * 0.5)
        raise FetchError(f"{what}: 重试 {self.cfg.retries} 次仍失败（{last}）")


def _main() -> int:
    import argparse

    from config import load_config, resolve
    from crawl.settings import build_settings
    from logging_setup import setup_logging

    ap = argparse.ArgumentParser(description="巨潮公告检索（只看元数据，不下载）")
    ap.add_argument("--year", default="2023", help="披露年份")
    ap.add_argument("--kind", default="annual", choices=sorted(CATEGORY))
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    setup_logging()
    cfg = build_settings(load_config(), resolve)
    client = CninfoClient(cfg)
    for ann in client.iter_announcements(args.kind, args.year, limit=args.limit):
        print(f"{ann.sec_code} {ann.sec_name:8s} {ann.year} {ann.size_kb:6d}KB {ann.title[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
