"""视觉大模型表格识别：解析与请求构造的离线用例。

不连网、不需要 GPU —— 只测"模型输出文本 → 单元格网格"这一段，因为线上
出问题八成是解析错位而不是模型不行。

    PYTHONPATH=src $PY tests/test_paddleocr_vl.py
    # 想打真服务时再给地址（默认不跑）
    PYTHONPATH=src $PY tests/test_paddleocr_vl.py --live http://127.0.0.1:8000/v1 \
        --model <模型名> --image work/crop.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.paddleocr_vl import (Cell, TableTextError, PaddleOcrVl, PaddleOcrVlError,  # noqa: E402
                             assign_bboxes, grid_problem, parse_table_text)


def check_grid(cells, n_rows: int, n_cols: int) -> tuple[int, int]:
    cover = [[0] * n_cols for _ in range(n_rows)]
    for c in cells:
        for dr in range(c.rowspan):
            for dc in range(c.colspan):
                r, q = c.row + dr, c.col + dc
                assert 0 <= r < n_rows and 0 <= q < n_cols, f"越界 {c}"
                cover[r][q] += 1
    holes = sum(1 for r in range(n_rows) for q in range(n_cols) if cover[r][q] == 0)
    dup = sum(1 for r in range(n_rows) for q in range(n_cols) if cover[r][q] > 1)
    return holes, dup


def test_html_span() -> None:
    text = """```html
    <table>
      <tr><td rowspan="2">项目</td><td colspan="2">2023 年度</td></tr>
      <tr><td>金额</td><td>占比</td></tr>
      <tr><td>货币资金</td><td>157,484,332,251.39</td><td>25.3%</td></tr>
    </table>
    ```"""
    cells, n_rows, n_cols = parse_table_text(text)
    assert (n_rows, n_cols) == (3, 3), (n_rows, n_cols)
    holes, dup = check_grid(cells, n_rows, n_cols)
    assert holes == 0 and dup == 0, (holes, dup)
    head = {(c.row, c.col): c for c in cells}
    assert head[(0, 0)].rowspan == 2, head[(0, 0)]
    assert head[(0, 1)].colspan == 2, head[(0, 1)]
    assert head[(2, 1)].text == "157,484,332,251.39", head[(2, 1)].text


def test_html_br_and_entities() -> None:
    cells, n_rows, n_cols = parse_table_text(
        "<table><tr><th>A&amp;B<br>第二行</th><td>&nbsp;</td></tr></table>")
    assert (n_rows, n_cols) == (1, 2)
    assert cells[0].text == "A&B\n第二行", repr(cells[0].text)
    assert cells[1].text == "", repr(cells[1].text)


def test_otsl() -> None:
    text = ("<fcel>项目<fcel>金额<fcel>占比<nl>"
            "<ucel><fcel>1,000<fcel>10%<nl>"
            "<fcel>合计<lcel><lcel><nl>")
    cells, n_rows, n_cols = parse_table_text(text)
    holes, dup = check_grid(cells, n_rows, n_cols)
    assert holes == 0 and dup == 0, (holes, dup)
    head = {(c.row, c.col): c for c in cells}
    assert head[(0, 0)].rowspan == 2, head[(0, 0)]
    assert head[(2, 0)].colspan == 3, head[(2, 0)]


def test_otsl_xcel() -> None:
    # 2×2 整块合并：第一行 A 横跨两列，第二行 ucel/xcel 两个标记都指向 A
    text = "<fcel>A<lcel><nl><ucel><xcel><nl>"
    cells, n_rows, n_cols = parse_table_text(text)
    assert (n_rows, n_cols) == (2, 2), (n_rows, n_cols)
    holes, dup = check_grid(cells, n_rows, n_cols)
    assert holes == 0 and dup == 0, (holes, dup)
    assert len(cells) == 1, cells
    assert (cells[0].rowspan, cells[0].colspan) == (2, 2), cells[0]


def test_html_without_closing_tags() -> None:
    # 模型经常省 </td>，隐式闭合没处理的话整行文字会挤进第一个格子
    cells, n_rows, n_cols = parse_table_text(
        "<table><tr><td>项目<td>金额<td>占比<tr><td>货币资金<td>1,000.00<td>25.3%</table>")
    assert (n_rows, n_cols) == (2, 3), (n_rows, n_cols)
    holes, dup = check_grid(cells, n_rows, n_cols)
    assert holes == 0 and dup == 0, (holes, dup)
    assert [c.text for c in cells if c.row == 1] == ["货币资金", "1,000.00", "25.3%"], cells


def test_grid_problem() -> None:
    ok = [Cell(0, 0, 1, 2, (0, 0, 0, 0), "a"), Cell(1, 0, 1, 2, (0, 0, 0, 0), "b")]
    assert grid_problem(ok, 2, 2) is None

    hole = [Cell(0, 0, 1, 1, (0, 0, 0, 0), "a")]           # 2x2 只给了一格
    assert "空洞" in grid_problem(hole, 2, 2)

    dup = [Cell(0, 0, 2, 2, (0, 0, 0, 0), "a"),           # a 已占满 2×2，b 又落进去
           Cell(1, 1, 1, 1, (0, 0, 0, 0), "b")]
    assert "重复覆盖" in grid_problem(dup, 2, 2)

    assert "太小" in grid_problem([Cell(0, 0, 1, 1, (0, 0, 0, 0), "a")], 1, 1)

    empty = [Cell(0, c, 1, 1, (0, 0, 0, 0), "") for c in range(10)]
    empty += [Cell(1, c, 1, 1, (0, 0, 0, 0), "") for c in range(10)]
    empty += [Cell(2, c, 1, 1, (0, 0, 0, 0), "x") for c in range(10)]
    assert "为空" in grid_problem(empty, 3, 10)


def test_bbox_and_unknown() -> None:
    cells, n_rows, n_cols = parse_table_text("<tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr>")
    boxed = assign_bboxes(cells, 100, 50)
    assert boxed[0].bbox == (0, 0, 50, 25), boxed[0].bbox
    assert boxed[-1].bbox == (50, 25, 100, 50), boxed[-1].bbox
    for bad in ("", "抱歉，我无法识别这张图片", "<table></table>"):
        try:
            parse_table_text(bad)
        except TableTextError:
            continue
        raise AssertionError(f"该报错却没报：{bad!r}")


def test_content_shapes() -> None:
    assert PaddleOcrVl._content({"choices": [{"message": {"content": " <table/> "}}]}) == " <table/> "
    assert PaddleOcrVl._content({"choices": [{"message": {"content": [
        {"type": "text", "text": "<table>"}, {"type": "text", "text": "</table>"}]}}]}) \
        == "<table></table>"
    for bad in ({}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}):
        try:
            PaddleOcrVl._content(bad)
        except PaddleOcrVlError:
            continue
        raise AssertionError(f"该报错却没报：{bad!r}")


def main(argv: list[str]) -> int:
    tests = [test_html_span, test_html_br_and_entities, test_html_without_closing_tags,
             test_otsl, test_otsl_xcel, test_grid_problem, test_bbox_and_unknown,
             test_content_shapes]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(tests)} 个用例通过")

    if "--live" in argv:
        import cv2
        import logging

        logging.basicConfig(level=logging.INFO)
        url = argv[argv.index("--live") + 1]
        model = argv[argv.index("--model") + 1] if "--model" in argv else ""
        image_path = Path(argv[argv.index("--image") + 1]) if "--image" in argv else None
        if image_path is None or not model:
            print("--live 需要同时给 --image 和 --model")
            return 2
        cfg = {"base_url": url, "model": model, "api_key": "EMPTY",
               "prompt": "识别这张表格，输出 HTML。", "max_tokens": 8192,
               "temperature": 0.0, "timeout": 180, "retries": 0, "max_side": 1600}
        image = cv2.imread(str(image_path))
        result = PaddleOcrVl(cfg).predict(image)
        holes, dup = check_grid(result.cells, result.n_rows, result.n_cols)
        print(f"live: {result.n_rows}×{result.n_cols}，{len(result.cells)} 格，"
              f"洞={holes} 重叠={dup}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
