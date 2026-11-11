#!/usr/bin/env python3
"""模型权重下载 / 校验。

仓库不含任何权重（models/ 约 2.2G，见 .gitignore），全部从 ModelScope 现取：

    python fetch_models.py list                  # 清单：分组 / 文件 / 大小 / 还缺哪些
    python fetch_models.py pull                  # 下载缺失项（大小已对上就跳过）
    python fetch_models.py pull --only vl,table  # 只下某几组：vl / text / layout / table / lore
    python fetch_models.py pull --force          # 忽略已有文件，重新下载
    python fetch_models.py verify                # 校验：文件大小 + sha256

只用标准库，Windows 与 WSL 都能跑。WSL 出网受限时在 Windows 侧执行，产物落 models/。
默认后端 paddleocr_vl 只需要 vl + text + layout 三组；table 与 lore 仅在 legacy 链路用到。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
BASE = "https://modelscope.cn/models/{repo}/resolve/master/{path}"
VL_REPO = "PaddlePaddle/PaddleOCR-VL-1.5"

# PaddleOCR-VL-1.5 权重：上游全量文件（跳过 .gitattributes 与评测结果）
VL_FILES = [
    ("LICENSE", 11376),
    ("README.md", 22955),
    ("added_tokens.json", 25381),
    ("chat_template.jinja", 1474),
    ("config.json", 2059),
    ("configuration_paddleocr_vl.py", 8104),
    ("generation_config.json", 133),
    ("image_processing_paddleocr_vl.py", 25032),
    ("inference.yml", 43),
    ("model.safetensors", 1917255968),
    ("modeling_paddleocr_vl.py", 110957),
    ("preprocessor_config.json", 641),
    ("processing_paddleocr_vl.py", 12253),
    ("processor_config.json", 137),
    ("special_tokens_map.json", 1151),
    ("tokenizer.json", 11189060),
    ("tokenizer.model", 1614363),
    ("tokenizer_config.json", 186947),
]

# 本地路径 ← 上游仓库里的文件。
# need: "core" = 默认后端就要；"legacy" = 只在 backend=legacy 时加载。
ENTRIES = [
    # ---- 文字检测 / 方向分类 / 识别 ----
    {
        "group": "text", "path": "ppocr/PP-OCRv6_det_small.onnx", "need": "core",
        "repo": "greatv/oar-ocr", "src": "pp-ocrv6_small_det.onnx",
        "size": 9880512, "alt": 9929594,
        "sha": "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
        "note": "PP-OCRv6 small 检测（官方权重的 ONNX 转换）",
    },
    {
        "group": "text", "path": "ppocr/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
        "need": "core", "repo": "greatv/oar-ocr",
        "src": "pp-lcnet_x0_25_textline_ori.onnx",
        "size": 1018629, "alt": 1018508,
        "sha": "fb402220f39b183d64a68cd48d4bd53267a21354b6fc39370c2a83fbdad85b10",
        "note": "文字行方向分类",
    },
    {
        "group": "text", "path": "ocr/PP-OCRv6_rec_small.onnx", "need": "core",
        "repo": "greatv/oar-ocr", "src": "pp-ocrv6_small_rec.onnx",
        "size": 21159378, "alt": 21234383,
        "sha": "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
        "note": "PP-OCRv6 small 识别",
    },
    {
        "group": "text", "path": "ocr/ppocrv6_dict.txt", "need": "core",
        "repo": "greatv/oar-ocr", "src": "ppocrv6_dict.txt",
        "size": 74947,
        "sha": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
        "note": "v6 识别字典",
    },
    {
        "group": "text", "path": "ocr/ppocr_keys_v1.txt", "need": "core",
        "repo": "greatv/oar-ocr", "src": "ppocr_keys_v1.txt",
        "size": 26250, "alt": 26249,
        "sha": "a1c84d9bdb9ab29043c58896224d32941783eb821629618416dcb08f12886492",
        "note": "v4 识别字典（备用识别模型用）",
    },
    # ---- 版面 ----
    {
        "group": "layout", "path": "layout/pp_doc_layoutv3.onnx", "need": "core",
        "repo": "PaddlePaddle/PP-DocLayoutV3_onnx", "src": "inference.onnx",
        "size": 130502049, "alt": 130502330,
        "sha": "45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba",
        "note": "PP-DocLayoutV3 官方 ONNX（上游文件名是 inference.onnx）",
    },
    # ---- legacy 链路 ----
    {
        "group": "table", "path": "table/unet.onnx", "need": "legacy",
        "repo": "RapidAI/RapidTable", "src": "unet.onnx",
        "size": 8335007,
        "sha": "0ea48d3a17e35ef5c2e498a5e799566073234d39b1079ca21d9f4fafe73c6d20",
        "note": "有线表格：线分割 UNet",
    },
    {
        "group": "table", "path": "table/table_cls.onnx", "need": "legacy",
        "repo": "RapidAI/RapidTable", "src": "table_cls/paddle_cls.onnx",
        "size": 6771838,
        "sha": "21c801f0c403cf960f9f1ccaecf506585b3b98421208033755b9e67cd2371492",
        "note": "有线 / 无线分类",
    },
    {
        "group": "lore", "path": "lore/pytorch_model.pt", "need": "legacy",
        "repo": "damo/cv_resnet-transformer_table-structure-recognition_lore",
        "src": "pytorch_model.pt",
        "size": 90543280,
        "sha": "6de715baad7360082d2fe367093a8af599e6dd98b1fcca4d832bdf63f2c66fad",
        "note": "读光 LORE 原始权重（转 ONNX 的输入）",
    },
    {
        "group": "lore", "path": "lore/configuration.json", "need": "legacy",
        "repo": "damo/cv_resnet-transformer_table-structure-recognition_lore",
        "src": "configuration.json",
        "size": 199,
        "sha": "450ef4d450cb270db877ce33cd420ed80487b704bec351758c6f0d3f2229f4f4",
        "note": "读光模型描述",
    },
]

# lore/det.onnx 与 lore/proc.onnx 是本项目从 pytorch_model.pt 导出的（IO 重命名成 img/feat），
# 上游 RapidTable 的 detect/process.onnx 输入签名不同，不能直接顶替，所以不列入自动下载。
ALT_NOTE = "部分条目多给一个可接受大小：开发机上是同模型的另一次导出（输入输出签名一致）。"

LORE_EXPORT = """
注意：lore/det.onnx 与 lore/proc.onnx 是项目内导出的 ONNX（输入已改名成 img / feat），
      上游 RapidTable 的同名文件输入签名不同、不能直接顶替。默认后端 paddleocr_vl 不需要
      它们；只有 backend=legacy 且遇到无线表格时才用。缺这两个文件时请改用默认后端。
"""

GROUPS = ("vl", "text", "layout", "table", "lore")


def accepted(entry: dict) -> tuple[int, ...]:
    """可接受的文件大小：上游值；部分模型开发机上是同模型的另一次导出，多给一个值。"""
    sizes = [entry["size"]]
    if entry.get("alt"):
        sizes.append(entry["alt"])
    return tuple(sizes)


def ready(entry: dict) -> bool:
    dest = MODELS / entry["path"]
    return dest.exists() and dest.stat().st_size in accepted(entry)


def human(n: int) -> str:
    return f"{n / 1048576:.1f}M" if n >= 1048576 else f"{n / 1024:.0f}K"


def url_of(repo: str, src: str) -> str:
    return BASE.format(repo=repo, path=src)


def tasks(groups=None):
    """展开成可执行的条目：paddleocr_vl 目录 + ENTRIES。"""
    picked = set(groups or GROUPS)
    out = []
    if "vl" in picked:
        for name, size in VL_FILES:
            out.append({"group": "vl", "path": f"paddleocr_vl/{name}", "need": "core",
                        "repo": VL_REPO, "src": name, "size": size, "sha": "",
                        "note": "PaddleOCR-VL-1.5"})
    out += [e for e in ENTRIES if e["group"] in picked]
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(entry: dict, force: bool = False, retries: int = 3) -> str:
    dest = MODELS / entry["path"]
    if ready(entry) and not force:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = url_of(entry["repo"], entry["src"])
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fetch_models/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as f:
                total = int(resp.headers.get("Content-Length") or entry["size"])
                got, mark = 0, time.time()
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if time.time() - mark > 2:
                        mark = time.time()
                        pct = f"{got * 100 / total:5.1f}%" if total else ""
                        print(f"\r      {human(got)} / {human(total)} {pct}", end="", flush=True)
                print("\r" + " " * 72 + "\r", end="")
            size = part.stat().st_size
            if size != entry["size"]:
                part.unlink(missing_ok=True)
                raise IOError(f"大小不对：{size} ≠ {entry['size']}")
            # 只有拿到的是上游原文件才比 sha256；等同模型的另一次导出时值本来就不同
            if entry.get("sha"):
                got_sha = sha256_of(part)
                if got_sha != entry["sha"]:
                    part.unlink(missing_ok=True)
                    raise IOError(f"sha256 不对：{got_sha[:12]} ≠ {entry['sha'][:12]}")
            part.replace(dest)
            return "ok"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt + 1 < retries:
                wait = 2 ** attempt
                print(f"    第 {attempt + 1} 次失败（{exc}），{wait}s 后重试")
                time.sleep(wait)
    raise SystemExit(f"下载失败 {entry['path']}：{last}")


def cmd_list(args) -> int:
    rows = tasks(args.only)
    print(f"{'分组':<7}{'文件':<52}{'大小':>8}  状态")
    missing = 0
    for e in rows:
        dest = MODELS / e["path"]
        if ready(e):
            state = "已就绪"
        elif dest.exists():
            state = f"大小不符（现 {human(dest.stat().st_size)}）"
            missing += 1
        else:
            state = "缺"
            missing += 1
        print(f"{e['group']:<7}{e['path']:<52}{human(e['size']):>8}  {state}")
    total = sum(e["size"] for e in rows)
    print(f"\n共 {len(rows)} 个文件 / {human(total)}，待下载 {missing} 个")
    if any(e.get("alt") for e in rows):
        print(ALT_NOTE)
    if {"lore"} & set(args.only or GROUPS):
        print(LORE_EXPORT)
    return 0


def cmd_pull(args) -> int:
    rows = tasks(args.only)
    done = skipped = 0
    for e in rows:
        dest = MODELS / e["path"]
        label = f"{e['path']:<52}"
        if ready(e) and not args.force:
            skipped += 1
            continue
        print(f"  ↓ {label}", end="", flush=True)
        state = download(e, force=args.force)
        print(f"  {human(e['size'])}  {'下载完成' if state == 'ok' else '已存在'}")
        done += state == "ok"
    print(f"\n完成：新下载 {done} 个，跳过 {skipped} 个 → {MODELS}")
    if "lore" in (args.only or GROUPS) and not all(
            (MODELS / "lore" / name).exists() for name in ("det.onnx", "proc.onnx")):
        print(LORE_EXPORT)
    return 0


def cmd_verify(args) -> int:
    rows = tasks(args.only)
    bad = []
    for e in rows:
        dest = MODELS / e["path"]
        if not dest.exists():
            bad.append((e["path"], "缺失"))
            continue
        size = dest.stat().st_size
        if size not in accepted(e):
            bad.append((e["path"], f"大小 {size} ≠ {e['size']}"))
            continue
        if e.get("sha") and size == e["size"] and sha256_of(dest) != e["sha"]:
            bad.append((e["path"], "sha256 不符"))
    if bad:
        print(f"× {len(bad)} 项不合格：")
        for path, why in bad:
            print(f"    {path:<52}{why}")
        return 1
    print(f"✓ {len(rows)} 个文件全部就绪（大小{'与 sha256' if any(e.get('sha') for e in rows) else ''}一致）")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="下载 / 校验 models/ 下的模型权重")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_text in (("list", cmd_list, "看清单与缺失项"),
                                ("pull", cmd_pull, "下载缺失项"),
                                ("verify", cmd_verify, "校验完整性")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--only", default="", help="只处理这些分组，逗号分隔：" + "/".join(GROUPS))
        if name == "pull":
            sp.add_argument("--force", action="store_true", help="忽略已有文件重新下载")
        sp.set_defaults(fn=fn)
    args = p.parse_args(argv)
    args.only = [g.strip() for g in (args.only or "").split(",") if g.strip()]
    unknown = [g for g in args.only if g not in GROUPS]
    if unknown:
        p.error(f"未知分组 {unknown}，可选：{', '.join(GROUPS)}")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
