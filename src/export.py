"""导出：整份文档 → json + excel。

json 按页面顺序原样写出（每页 regions[] 里是 table / char 两类分区）。

excel **一页一个 sheet**：把该页的分区按上下位置铺成一张大表格，
表格分区画粗框线并还原合并单元格，文字分区照原文一行一行占满整行、不加框线。
"""

from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

# 表格线要"明显"：中等粗细 + 深色，别用浅灰细线
GRID = Border(left=Side(style="medium", color="404040"),
              right=Side(style="medium", color="404040"),
              top=Side(style="medium", color="404040"),
              bottom=Side(style="medium", color="404040"))


def write_json(doc: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _grid_cols(page: dict) -> int:
    """这一页大表用几列：取最大那张表的列数，没有表就 1 列。"""
    return max([r["n_cols"] for r in page.get("regions", [])
                if r["type"] == "table"] or [1])


def write_excel(doc: dict, path: Path) -> Path:
    wb = Workbook()
    wb.remove(wb.active)

    for page in doc["pages"]:
        ws = wb.create_sheet(f"p{page['page']}"[:31])
        n_cols = _grid_cols(page)
        cursor = 1

        for region in page.get("regions", []):
            if region["type"] == "table":
                # 先给整块网格铺满边框，再写字、再合并：否则合并后从属格子
                # 没有边框，表格会出现缺口（合并区域只有左上角带样式）
                for dr in range(region["n_rows"]):
                    for dc in range(region["n_cols"]):
                        ws.cell(row=cursor + dr, column=dc + 1).border = GRID
                for cell in region["cells"]:
                    row = cursor + cell["row"]
                    col = cell["col"] + 1
                    target = ws.cell(row=row, column=col, value=cell["ocr"])
                    target.alignment = Alignment(vertical="center", wrap_text=True)
                    target.border = GRID
                    if cell["rowspan"] > 1 or cell["colspan"] > 1:
                        ws.merge_cells(start_row=row, start_column=col,
                                       end_row=row + cell["rowspan"] - 1,
                                       end_column=col + cell["colspan"] - 1)
                cursor += region["n_rows"]
            else:                                   # 文字分区：占满整行，不画框
                for line in region["lines"]:
                    target = ws.cell(row=cursor, column=1, value=line["text"])
                    target.alignment = Alignment(vertical="center", wrap_text=True)
                    if n_cols > 1:
                        ws.merge_cells(start_row=cursor, start_column=1,
                                       end_row=cursor, end_column=n_cols)
                    cursor += 1
            cursor += 1                             # 分区之间空一行

        width = 100 if n_cols == 1 else 20
        for col in range(1, n_cols + 1):
            ws.column_dimensions[get_column_letter(col)].width = width

        if cursor == 1:                             # 整页什么都没识别出来
            ws.cell(row=1, column=1, value=f"第 {page['page']} 页：无内容").font = Font(bold=True)

    if not wb.sheetnames:                           # 空文档也要能打开
        wb.create_sheet("空")

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
