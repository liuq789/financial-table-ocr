#!/usr/bin/env bash
# 起一个 OpenAI 兼容的视觉大模型服务（默认 PaddleOCR-VL-1.5）。
#
# 为什么不直接 vllm serve：本环境 `openai==0.28`，vLLM 0.19 一导入就缺
# `openai.types`。新 openai 装在隔离目录 `~/vllm_extra`（不污染 conda env），
# 这里把它放到 PYTHONPATH 最前面即可。
#
#   bash serve_model.sh                       # 默认 8000 端口
#   PORT=8001 MODEL=models/paddleocr_vl bash serve_model.sh
#   VLLM_EXTRA=/path/to/dir bash serve_model.sh   # 换了隔离目录
#
# 启动前会先查显存：不够就直接退出并说明差多少，不会让 vLLM 半路 OOM
# （显存被别的进程占着时，vLLM 会在加载 KV cache 时崩，日志很难看懂）。
#
# 起完验证：curl -s localhost:8000/v1/models
set -euo pipefail

cd "$(dirname "$0")"          # 脚本就在项目根目录

# 选 Python：优先 $PY，其次 conda env "llm"（本机就是这种），最后系统 python3
if [ -z "${PY:-}" ]; then
  for cand in "$HOME/miniconda3/envs/llm/bin/python" \
              "$HOME/anaconda3/envs/llm/bin/python" \
              "$HOME/miniforge3/envs/llm/bin/python"; do
    [ -x "$cand" ] && { PY=$cand; break; }
  done
fi
PY=${PY:-$(command -v python3 || true)}
[ -n "$PY" ] && [ -x "$PY" ] || { echo "找不到 Python，用 PY=/path/to/python 指定" >&2; exit 1; }
VLLM=${VLLM:-$(dirname "$PY")/vllm}
MODEL=${MODEL:-models/paddleocr_vl}
MODEL_NAME=${MODEL_NAME:-PaddleOCR-VL}   # 别用 NAME：系统里可能已有同名环境变量
PORT=${PORT:-8000}
GPU_MEM=${GPU_MEM:-0.60}          # 留给同机的 ONNX 版面/结构模型
MAX_LEN=${MAX_LEN:-16384}
EAGER=${EAGER:-1}                 # 1=跳过 torch.compile（WSL 上编译要 5~10 分钟）

VLLM_EXTRA=${VLLM_EXTRA:-$HOME/vllm_extra}
if [ -d "$VLLM_EXTRA" ]; then
  export PYTHONPATH="$VLLM_EXTRA${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "缺少隔离依赖目录 $VLLM_EXTRA（vLLM 需要 openai>=1.x）" >&2
  echo "装法见 README「部署模型服务」/\"Serving the model\"" >&2
  exit 1
fi

# ---- 显存预检：留够余量再启动，避免 vLLM 加载到一半 OOM ----
NEED_MB=${NEED_MB:-8192}          # 权重 1.8G + KV 4.5G + 激活，8G 起步
if command -v nvidia-smi >/dev/null 2>&1; then
  read -r TOTAL FREE <<<"$(nvidia-smi --query-gpu=memory.total,memory.free \
      --format=csv,noheader,nounits | head -1 | tr -d ',')"
  echo "显存 空闲 ${FREE}MB / 总 ${TOTAL}MB（需要 ≥${NEED_MB}MB）"
  if [ "${FREE:-0}" -lt "$NEED_MB" ]; then
    echo "显存不足：只有 ${FREE}MB 可用，至少需要 ${NEED_MB}MB。" >&2
    echo "先看谁在占：nvidia-smi；或把 NEED_MB 调小/改 GPU_MEM 再试。" >&2
    exit 2
  fi
else
  echo "没有 nvidia-smi，跳过显存预检（如果是 CPU 机器，vLLM 跑不起来）" >&2
fi

echo "模型 $MODEL  名字 $MODEL_NAME  端口 $PORT  GPU 占比 $GPU_MEM  eager $EAGER"
ARGS=(--served-model-name "$MODEL_NAME" --trust-remote-code
      --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_LEN"
      --gpu-memory-utilization "$GPU_MEM" --limit-mm-per-prompt '{"image":1}')
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)

# ---------------------------------------------------------------- 守护重启
# 模型服务崩了（OOM、CUDA 错误、偶发段错误）不能就这么死着——上层虽然有
# fallback_legacy 兜底，但那一路的准确率差一档。这里前台守护：非正常退出就重启，
# 每次重启前重新做显存预检（上次很可能是显存不够），退避递增，连续失败太多次才放弃。
RESTART_MAX=${RESTART_MAX:-5}     # 连续失败上限（成功跑起来后计数会被清零）
RESTART_WAIT=${RESTART_WAIT:-5}   # 首次重启等待秒数，之后翻倍
fail=0
while true; do
  started=$SECONDS
  set +e
  "$VLLM" serve "$MODEL" "${ARGS[@]}"
  code=$?
  set -e
  # 跑够 2 分钟就算"站住了"：之后偶发崩不累积失败次数，避免偶发问题被算成连续失败
  if [ $((SECONDS - started)) -gt 120 ]; then
    fail=0
  fi
  if [ "$code" -eq 0 ]; then
    echo "模型服务正常退出（code 0）"
    exit 0
  fi
  fail=$((fail + 1))
  if [ "$fail" -gt "$RESTART_MAX" ]; then
    echo "连续重启 ${RESTART_MAX} 次仍失败（最后 code=$code），放弃。检查 work/vllm_serve.log" >&2
    exit "$code"
  fi
  wait_s=$((RESTART_WAIT * fail))
  echo "[$(date '+%H:%M:%S')] 模型服务异常退出 code=$code，第 $fail 次重启，${wait_s}s 后" >&2
  sleep "$wait_s"
  # 重启前再看一眼显存：上次崩很可能就是显存被占满了
  if command -v nvidia-smi >/dev/null 2>&1; then
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "[$(date '+%H:%M:%S')] 重启前显存空闲 ${FREE}MB"
  fi
done
