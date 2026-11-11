#!/usr/bin/env bash
# 一键拉起整套服务：PaddleOCR-VL（vLLM）+ 网页前端。
#
#   bash start.sh                     # 起模型 + 前端，等到就绪再打印地址
#   bash start.sh status              # 两个服务在不在
#   bash start.sh stop                # 全停（连 vLLM 的子进程一起收）
#   bash start.sh restart             # 重起
#   bash start.sh logs                # 跟踪两个日志
#   bash start.sh start --no-model    # 只起前端（模型挂了也能用，质量低一档）
#
# 环境变量：WEB_PORT(8010) / MODEL_PORT(8000) / PY / GPU_MEM / MAX_LEN / EAGER
#   WEB_PORT=8020 bash start.sh
#   GPU_MEM=0.5 bash start.sh
#
# 原则：**模型起不来也要把前端拉起来**。前端有 fallback_legacy，会退回本地 ONNX，
# 只是准确率低一档——结尾会明确说明当前是哪种模式，不让人误以为模型在跑。
set -uo pipefail

cd "$(dirname "$0")" || exit 1
ROOT=$PWD
RUN=$ROOT/work/run
LOGS=$ROOT/work/logs
WEB_PORT=${WEB_PORT:-8010}
MODEL_PORT=${MODEL_PORT:-8000}
# 选 Python：优先 $PY，其次 conda env "llm"（本机就是这种），最后系统 python3
if [ -z "${PY:-}" ]; then
  for cand in "$HOME/miniconda3/envs/llm/bin/python" \
              "$HOME/anaconda3/envs/llm/bin/python" \
              "$HOME/miniforge3/envs/llm/bin/python"; do
    [ -x "$cand" ] && { PY=$cand; break; }
  done
fi
PY=${PY:-$(command -v python3 || true)}
WAIT_MODEL=${WAIT_MODEL:-180}
WAIT_WEB=${WAIT_WEB:-30}

mkdir -p "$RUN" "$LOGS"

# ---------------------------------------------------------------- 输出
if [ -t 1 ]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; D=""; N=""
fi
ok()   { printf '%s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '%s✗%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
step() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }

# ---------------------------------------------------------------- 工具
pid_of() { [ -f "$1" ] && cat "$1" 2>/dev/null; }

alive() {                                 # alive <pid文件>
  local pid; pid=$(pid_of "$1")
  [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null
}

http_ok() {                               # http_ok <url> [超时秒]
  curl -fsS -m "${2:-3}" "$1" >/dev/null 2>&1
}

port_pid() {                              # 谁占着这个端口（没有则空）
  ss -ltnp 2>/dev/null | awk -v p=":$1" '$4 ~ p"$" {print $NF}' |
    sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1
}

wait_http() {                             # wait_http <url> <秒> <阶段名>
  local url=$1 limit=$2 name=$3 waited=0
  while [ "$waited" -lt "$limit" ]; do
    http_ok "$url" 2 && return 0
    sleep 2
    waited=$((waited + 2))
    if [ $((waited % 20)) -eq 0 ]; then
      printf '  %s%s… %ds%s\n' "$D" "$name" "$waited" "$N"
    fi
  done
  return 1
}

spawn() {                                 # spawn <pid文件> <日志> <命令…>
  local pidfile=$1 logfile=$2
  shift 2
  # setsid 开新会话：脚本退出/关终端不影响它；$$ 就是新会话的首进程，
  # 于是 kill -TERM -- -PID 能一次性收掉它和它的子进程（vLLM 会 fork 引擎）。
  nohup setsid bash -c 'echo $$ >"$1"; shift; exec "$@"' _ "$pidfile" "$@" \
      >>"$logfile" 2>&1 &
  sleep 0.3
  [ -s "$pidfile" ] || die "启动失败：$pidfile 没写出来（看 $logfile）"
}

# ---------------------------------------------------------------- 体检
preflight() {
  step "1/4 环境检查"
  MODEL_UP=0
  WEB_UP=0
  http_ok "http://127.0.0.1:$MODEL_PORT/v1/models" 2 && MODEL_UP=1
  http_ok "http://127.0.0.1:$WEB_PORT/api/health" 2 && WEB_UP=1

  [ -n "$PY" ] && [ -x "$PY" ] ||     die "找不到 Python，用 PY=/path/to/python 指定（如 conda env 的 .../envs/llm/bin/python）"
  ok "Python $("$PY" -V 2>&1 | awk '{print $2}')　$PY"

  [ -s "$ROOT/models/paddleocr_vl/model.safetensors" ] || \
    die "缺模型权重（1.8G）。先跑：python fetch_models.py pull --only vl"
  ok "模型权重 $(du -sh "$ROOT/models/paddleocr_vl" 2>/dev/null | cut -f1)"
  if [ ! -f "$ROOT/models/lore/det.onnx" ]; then
    warn "缺 legacy 链路的模型（models/lore、models/table）；默认后端不受影响，见 README「模型下载」"
  fi

  [ -f "$ROOT/serve_model.sh" ] || die "缺 serve_model.sh"
  [ -f "$ROOT/src/web/server.py" ] || die "缺 src/web/server.py"

  if [ "$NO_MODEL" != "1" ]; then
    local extra=${VLLM_EXTRA:-$HOME/vllm_extra}
    [ -d "$extra" ] || die "缺隔离依赖 $extra（vLLM 0.19 需要 openai>=1.x，见 README）"
    ok "隔离依赖 $extra"
    "$PY" -c "import vllm" 2>/dev/null || die "vLLM 导入失败，试：$PY -c 'import vllm'"
    ok "vLLM $("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

    if [ "$MODEL_UP" = "1" ]; then
      ok "模型服务已在运行，显存由它占着（跳过显存检查）"
    elif command -v nvidia-smi >/dev/null; then
      local free
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      if [ "${free:-0}" -lt 8192 ]; then
        warn "显存只剩 ${free}MB（需要 ≥8192MB），模型很可能起不来；先 nvidia-smi 看看谁在占"
      else
        ok "显存空闲 ${free}MB"
      fi
    else
      warn "没有 nvidia-smi：GPU 不可用，模型起不来（前端仍能用 legacy）"
    fi
  fi

  local spec port name pid
  for spec in "$MODEL_PORT:模型服务" "$WEB_PORT:前端"; do
    port=${spec%%:*}; name=${spec#*:}
    pid=$(port_pid "$port")
    if { [ "$port" = "$MODEL_PORT" ] && [ "$MODEL_UP" = "1" ]; } ||
       { [ "$port" = "$WEB_PORT" ] && [ "$WEB_UP" = "1" ]; }; then
      ok "端口 $port 上就是本服务（pid=$pid，已就绪）"
    elif [ -n "${pid:-}" ]; then
      warn "端口 $port（$name）被别的进程 pid=$pid 占着；先确认是不是上次没停干净（bash start.sh status）"
    else
      ok "端口 $port 空闲（$name）"
    fi
  done
}

# ---------------------------------------------------------------- 起服务
start_model() {
  step "2/4 模型服务（PaddleOCR-VL + vLLM，端口 $MODEL_PORT）"
  if http_ok "http://127.0.0.1:$MODEL_PORT/v1/models" 2; then
    local pid; pid=$(port_pid "$MODEL_PORT")
    [ -n "${pid:-}" ] && echo "$pid" >"$RUN/model.pid"   # 记下来，stop 才杀得掉
    ok "已经在跑，跳过（pid=${pid:-?}）"
    return 0
  fi
  if alive "$RUN/model.pid"; then
    warn "进程在（pid=$(pid_of "$RUN/model.pid")），等它加载完"
  else
    : >"$LOGS/model.out"
    spawn "$RUN/model.pid" "$LOGS/model.out" \
          env PORT="$MODEL_PORT" bash "$ROOT/serve_model.sh"
    ok "已拉起 pid=$(pid_of "$RUN/model.pid")，日志 $LOGS/model.out"
  fi

  if wait_http "http://127.0.0.1:$MODEL_PORT/v1/models" "$WAIT_MODEL" "模型加载"; then
    local ids
    ids=$(curl -fsS -m 3 "http://127.0.0.1:$MODEL_PORT/v1/models" |
          "$PY" -c 'import sys, json; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)
    ok "模型就绪：${ids:-?}"
    return 0
  fi

  warn "模型 ${WAIT_MODEL}s 内未就绪，日志末尾："
  tail -15 "$LOGS/model.out" 2>/dev/null | sed 's/^/    /'
  return 1
}

start_web() {
  step "3/4 前端（端口 $WEB_PORT）"
  if http_ok "http://127.0.0.1:$WEB_PORT/api/health" 2; then
    local pid; pid=$(port_pid "$WEB_PORT")
    [ -n "${pid:-}" ] && echo "$pid" >"$RUN/web.pid"     # 同上
    ok "已经在跑，跳过（pid=${pid:-?}）"
    return 0
  fi
  if alive "$RUN/web.pid"; then
    kill -TERM -- "-$(pid_of "$RUN/web.pid")" 2>/dev/null
    sleep 1
  fi
  : >"$LOGS/web.out"
  spawn "$RUN/web.pid" "$LOGS/web.out" \
        env PYTHONPATH="$ROOT/src" "$PY" -u -m web.server --port "$WEB_PORT"
  ok "已拉起 pid=$(pid_of "$RUN/web.pid")，日志 $LOGS/web.out"

  if wait_http "http://127.0.0.1:$WEB_PORT/api/health" "$WAIT_WEB" "前端启动"; then
    ok "前端就绪"
    return 0
  fi
  warn "前端 ${WAIT_WEB}s 内未就绪，日志末尾："
  tail -15 "$LOGS/web.out" 2>/dev/null | sed 's/^/    /'
  return 1
}

report_health() {
  local payload
  payload=$(curl -fsS -m 3 "http://127.0.0.1:$WEB_PORT/api/health" 2>/dev/null) || return 0
  [ -n "$payload" ] || return 0
  "$PY" - "$payload" <<'PY' 2>/dev/null
import json
import sys

data = json.loads(sys.argv[1])
mem = data.get("memory") or {}
gpu = data.get("gpu") or {}
svc = data.get("model_service") or {}
gate = data.get("gate") or {}
print(f"          可用内存 {mem.get('available_mb', '?')}MB"
      f" · 本进程 {data.get('rss_mb', '?')}MB"
      f" · 显存空闲 {gpu.get('free_mb', '-')}MB")
print(f"          模型服务 {'在线' if svc.get('ok') else '离线'}"
      f" · 模型名匹配 {svc.get('matched')}"
      f" · 并发 {gate.get('max_inflight', '?')}+排队 {gate.get('max_pending', '?')}")
PY
}

report() {
  step "4/4 状态"
  local model_ok=0 web_ok=0
  http_ok "http://127.0.0.1:$MODEL_PORT/v1/models" 2 && model_ok=1
  http_ok "http://127.0.0.1:$WEB_PORT/api/health" 2 && web_ok=1

  if [ "$model_ok" = 1 ]; then
    ok "模型服务  http://127.0.0.1:$MODEL_PORT/v1　pid=$(pid_of "$RUN/model.pid")"
  else
    warn "模型服务  未就绪 —— 前端会自动退回本地 ONNX（准确率低一档）"
  fi

  if [ "$web_ok" = 1 ]; then
    ok "前端      http://127.0.0.1:$WEB_PORT　pid=$(pid_of "$RUN/web.pid")"
    report_health
  else
    warn "前端      未就绪"
  fi

  printf '\n%s常用命令%s\n' "$B" "$N"
  printf '  打开        http://127.0.0.1:%s\n' "$WEB_PORT"
  printf '  看日志      bash start.sh logs\n'
  printf '  全停        bash start.sh stop\n'
  printf '  仅前端      bash start.sh start --no-model\n'
}

stop_all() {
  step "停止服务"
  local spec name file pid
  for spec in "前端:$RUN/web.pid" "模型:$RUN/model.pid"; do
    name=${spec%%:*}; file=${spec#*:}
    pid=$(pid_of "$file")
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      local i=0
      while [ "$i" -lt 20 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 0.5; i=$((i + 1))
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
        warn "$name 强制结束（pid=$pid）"
      else
        ok "$name 已停（pid=$pid）"
      fi
    else
      printf '  %s 没在跑\n' "$name"
    fi
    rm -f "$file"
  done

  # 兜底：pid 文件丢了（例如手工起的、或换了终端）就按端口找，
  # 但只杀 cmdline 里带本项目特征的进程，别人的进程绝不碰。
  local port cmd
  for port in "$WEB_PORT" "$MODEL_PORT"; do
    pid=$(port_pid "$port")
    [ -n "${pid:-}" ] || continue
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)
    case "$cmd" in
      *web.server*|*vllm*|*serve_model.sh*)
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        sleep 1
        kill -0 "$pid" 2>/dev/null && { kill -KILL "$pid" 2>/dev/null; }
        ok "端口 $port 上的残留进程已清理（pid=$pid）" ;;
      "")
        : ;;
      *)
        warn "端口 $port 被别的进程占着（pid=$pid）：${cmd:0:60}… 没动它" ;;
    esac
  done
}

status_all() {
  local model_ok=0 web_ok=0
  http_ok "http://127.0.0.1:$MODEL_PORT/v1/models" 2 && model_ok=1
  http_ok "http://127.0.0.1:$WEB_PORT/api/health" 2 && web_ok=1
  if [ "$model_ok" = 1 ]; then
    ok "模型  在  http://127.0.0.1:$MODEL_PORT/v1"
  else
    warn "模型  停"
  fi
  if [ "$web_ok" = 1 ]; then
    ok "前端  在  http://127.0.0.1:$WEB_PORT"
  else
    warn "前端  停"
  fi
  [ "$model_ok" = 1 ] && [ "$web_ok" = 1 ]
}

# ---------------------------------------------------------------- 入口
ACTION=${1:-start}
[ "$#" -gt 0 ] && shift
NO_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --no-model) NO_MODEL=1 ;;
    *) ;;
  esac
done

case "$ACTION" in
  start)
    preflight
    if [ "$NO_MODEL" = "1" ]; then
      warn "按 --no-model 跳过模型服务：前端只会走本地 ONNX 链路"
    else
      start_model || true           # 模型失败不挡前端
    fi
    start_web || die "前端起不来，先看 $LOGS/web.out"
    report
    ;;
  stop|down)  stop_all ;;
  restart)    stop_all; sleep 1; exec bash "$0" start "$@" ;;
  status|st)  status_all ;;
  logs)       tail -n 40 -F "$LOGS/model.out" "$LOGS/web.out" ;;
  *)
    printf '用法：bash start.sh [start|stop|restart|status|logs] [--no-model]\n'
    exit 1 ;;
esac
