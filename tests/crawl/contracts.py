"""抓取侧的数据契约。跨模块只传这里定义的类型，不传裸 dict。

三类东西：

    Announcement   一条公告（巨潮口径），只描述"要下什么"，不含任何网络细节
    PageHit        一页的打分结果，决定"这页要不要留"
    RenderedPage   已经渲出来的页图（PNG 字节），存储层直接落盘

单位约定：bbox 一律是 PDF 点坐标 (x0, y0, x1, y1)，左上原点，y 向下；
尺寸单位 pt；字节单位 byte；时间 ISO-8601 字符串（UTC）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Announcement:
    """一条待抓公告。key 是去重主键：(证券代码, 报告年度, 类型)。"""

    sec_code: str
    sec_name: str
    title: str
    year: str            # 报告年度，四位数字符串；解析不出时为空串
    kind: str            # annual | semiannual
    url: str             # PDF 直链
    size_kb: int = 0
    published: str = ""  # 公告日期 YYYY-MM-DD

    @property
    def key(self) -> str:
        return f"{self.sec_code}_{self.year}_{self.kind}"

    @property
    def slug(self) -> str:
        """落盘名：000651_2023_annual。不含扩展名。"""
        return f"{self.sec_code}_{self.year}_{self.kind}"


@dataclass(frozen=True)
class PageHit:
    """一页的筛选结果。

    rows      判定为"数据行"的行数（同一基线上 >=2 个数字 token 且总 token >=3）
    max_cols  单行最多几个数字 token，粗估列数
    boxes     数据行按纵向空白分块后的候选表格框（可能多块），只用于粗定位
    """

    page_no: int          # 1 起
    rows: int
    max_cols: int
    chars: int
    boxes: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def n_cols(self) -> int:
        return max(self.max_cols + 1, 2)   # 数字列 + 1 列标签

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """全页候选框的并集；没有块时为 None。"""
        if not self.boxes:
            return None
        return (
            min(b[0] for b in self.boxes),
            min(b[1] for b in self.boxes),
            max(b[2] for b in self.boxes),
            max(b[3] for b in self.boxes),
        )


@dataclass(frozen=True)
class RenderedPage:
    """一页渲染结果。png 是内存里的字节，由存储层决定落到哪。"""

    hit: PageHit
    png: bytes
    width: int
    height: int
    dpi: int


@dataclass(frozen=True)
class DocOutcome:
    """一份 PDF 处理完的结果，供台账记账与日志。"""

    ann: Announcement
    n_pages: int = 0
    pages: list[RenderedPage] = field(default_factory=list)
    elapsed_s: float = 0.0
    kept_pdf: bool = False
    pdf_bytes: int = 0

    @property
    def n_images(self) -> int:
        return len(self.pages)

    @property
    def image_bytes(self) -> int:
        return sum(len(p.png) for p in self.pages)


@dataclass(frozen=True)
class CrawlSettings:
    """crawl 段的全部参数。由 config.yaml 组装，代码里不写死。"""

    data_root: Path
    workers: int = 4
    page_dpi: int = 200
    min_rows: int = 5            # 少于此行数不算表格页
    max_pages_per_doc: int = 8   # 单份报告最多留几页（优先留数据行最多的）
    max_pdf_mb: int = 60
    pdf_budget_gb: float = 60.0  # PDF 原件总预算，超了就不再保留
    min_free_gb: float = 20.0    # 磁盘保底线，低于它整个任务停下
    target_images: int = 30000
    max_docs: int = 0            # 0 = 不限
    timeout_s: float = 60.0
    retries: int = 3
    retry_wait_s: float = 2.0
    min_delay_s: float = 0.15

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def pages_dir(self) -> Path:
        return self.data_root / "pages"

    @property
    def db_path(self) -> Path:
        return self.data_root / "crawl.sqlite"
