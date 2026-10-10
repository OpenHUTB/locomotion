#!/usr/bin/env bash
# =====================================================================
# main.sh —— qflex_benchmark 的唯一入口
#
#   bash main.sh [命令] [选项...]
#
# 命令（不给则显示帮助）：
#   check    只做前置检查（命令/GPU/内存/磁盘），不改动任何东西
#   setup    克隆 Qflex + 打两个补丁 + 建 4 套 venv            约 40~60 min
#   verify   校验 7 个环境能建起来，打印 |S|/|A|（对论文 Fig.2(a)）
#   train    跑批 6 环境 × 6 算法 × 3 种子 = 108 run          约 300 h
#   plot     出图（六格学习曲线 + per-seed CSV）
#   all      按顺序执行 setup → verify → train → plot
#
# 选项：
#   -r, --root DIR       Qflex 源码目录          默认 ~/work/Qflex
#   -s, --seeds "100 101 102"                    默认这三个
#       --seed N         只跑一个种子（等价 -s N）
#       --no-ostrich     不跑 Ostrich（5 环境 / 90 run，内存小的机器用）
#       --only ENV_ID    只跑指定环境        --skip ENV_ID  跳过指定环境
#       --algs "qflex sac crossq qsm sdac dacer"
#       --watch-mb N     看门狗水位；默认留给底层脚本按环境自动决定
#   -o, --out DIR        图与 CSV 输出目录       默认 <本目录>/figs
#   -n, --dry-run        干跑：只打印将要执行的命令，不占 GPU、不写产物
#   -y, --yes            跳过运行前确认
#       --no-tmux        不要自建 tmux 会话（你自己已经在一个里面）
#   -h, --help
#
# 长阶段（setup / train / all）会自动放进 tmux 会话 qflex_main，
# 日志在 ~/qflex_logs/main_<时间戳>.log；中途断了原样重跑即可，
# 已完成的 run 会被跳过（各 run 目录里有 .formal_done 标记）。
#
# 环境变量与选项等价，可直接 export（QROOT / SEEDS / WITH_OSTRICH / ONLY /
# SKIP / ALGS / WATCH_MB / DRYRUN ...），选项优先级更高。
# =====================================================================
set -uo pipefail

SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)/$(basename -- "${BASH_SOURCE[0]:-$0}")"
HERE="$(dirname -- "$SELF")"
QROOT="${QROOT:-$HOME/work/Qflex}"
SEEDS="${SEEDS:-100 101 102}"
WITH_OSTRICH="${WITH_OSTRICH:-1}"          # 默认跑满 6 格
ALGS="${ALGS:-}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
WATCH_MB="${WATCH_MB:-}"
OUT="${OUT:-$HERE/figs}"
LOGDIR="${LOGDIR:-$HOME/qflex_logs}"
SESSION="${SESSION:-qflex_main}"
DRYRUN="${DRYRUN:-0}"
NO_TMUX="${NO_TMUX:-0}"
ASSUME_YES=0
CMD=""

C_TI=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_NO=$'\033[1;31m'; C_WA=$'\033[1;33m'; C_0=$'\033[0m'
say(){  printf '\n%s== %s%s\n' "$C_TI" "$1" "$C_0"; }
ok(){   printf '%s[OK]%s   %s\n'   "$C_OK" "$C_0" "$1"; }
no(){   printf '%s[FAIL]%s %s\n'   "$C_NO" "$C_0" "$1"; }
warn(){ printf '%s[WARN]%s %s\n'   "$C_WA" "$C_0" "$1"; }
info(){ printf '       %s\n' "$1"; }

usage(){ sed -n '2,35p' "$SELF" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------- 解析参数
while [ $# -gt 0 ]; do
  case "$1" in
    check|setup|verify|train|plot|all) CMD="$1"; shift ;;
    -r|--root)     QROOT="$2"; shift 2 ;;
    -s|--seeds)    SEEDS="$2"; shift 2 ;;
    --seed)        SEEDS="$2"; shift 2 ;;
    --no-ostrich)  WITH_OSTRICH=0; shift ;;
    --ostrich)     WITH_OSTRICH=1; shift ;;
    --only)        ONLY="$2"; shift 2 ;;
    --skip)        SKIP="$2"; shift 2 ;;
    --algs)        ALGS="$2"; shift 2 ;;
    --watch-mb)    WATCH_MB="$2"; shift 2 ;;
    -o|--out)      OUT="$2"; shift 2 ;;
    -n|--dry-run)  DRYRUN=1; shift ;;
    -y|--yes)      ASSUME_YES=1; shift ;;
    --no-tmux)     NO_TMUX=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             no "未知参数：$1"; usage; exit 2 ;;
  esac
done
[ -n "$CMD" ] || CMD="help"
[ "$CMD" = "help" ] && { usage; exit 0; }

export QROOT SEEDS WITH_OSTRICH ALGS ONLY SKIP WATCH_MB OUT LOGDIR DRYRUN NO_TMUX SESSION
if [ "$DRYRUN" = "1" ]; then export DRYRUN=1; else export DRYRUN=0; fi

# 需要放进 tmux 的长阶段
is_long(){ case "$1" in setup|train|all) return 0 ;; *) return 1 ;; esac; }

NENV=5; [ "$WITH_OSTRICH" = "1" ] && NENV=6
NRUN=$(( NENV * 6 ))

# ---------------------------------------------------------------- 计划与成本
plan_cost(){
  local nseed hours
  nseed="$(printf '%s\n' $SEEDS | grep -c . || echo 0)"
  if [ "$WITH_OSTRICH" = "1" ]; then
    hours=101
    info "环境数：6（含 Ostrich，单 run 进程峰值约 24.5 GB）"
  else
    hours=73
    info "环境数：5（不含 Ostrich；要第 6 格请去掉 --no-ostrich）"
  fi
  info "单种子：约 ${hours} h（dacer 占其中近一半）"
  info "本次 $nseed 个种子 ⇒ 串行约 $((nseed * hours)) h ≈ $((nseed * hours / 24)) 天"
}

print_plan(){
  say "计划"
  info "入口    ：$SELF"
  info "命令    ：$CMD"
  info "Qflex   ：$QROOT"
  info "种子    ：[$SEEDS]"
  info "算法    ：${ALGS:-qflex sac crossq qsm sdac dacer（默认）}"
  [ -n "$ONLY" ] && info "只跑    ：$ONLY"
  [ -n "$SKIP" ] && info "跳过    ：$SKIP"
  info "出图到  ：$OUT"
  info "日志    ：$LOGDIR"
  info "Ostrich ：$([ "$WITH_OSTRICH" = 1 ] && echo '启用（6 环境 / 每种子 36 run）' || echo '未启用（5 环境 / 每种子 30 run）')"
  [ "$DRYRUN" = "1" ] && info "干跑    ：是（不占 GPU、不写产物）"
}

# ---------------------------------------------------------------- 阶段 1：check
stage_check(){
  say "前置检查"
  local bad=0

  if [ "$(uname -s)" = "Linux" ]; then ok "系统：$(uname -sr)"; else warn "非 Linux（$(uname -s)）；本流水线只在 WSL2 Ubuntu 上实测过"; fi

  if command -v tmux >/dev/null 2>&1; then ok "tmux：$(tmux -V)"; else no "缺少 tmux（sudo apt-get install -y tmux）"; bad=1; fi
  if command -v git  >/dev/null 2>&1; then ok "git ：$(git --version | awk '{print $3}')"; else no "缺少 git"; bad=1; fi
  if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then ok "curl/wget：有"; else no "缺少 curl/wget（setup 需要联网下载）"; bad=1; fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU ：$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  else
    warn "没有 nvidia-smi：训练会退回 CPU，基本不可行"
  fi

  if [ -r /proc/meminfo ]; then
    local tot av
    tot="$(awk '/^MemTotal:/{printf "%.1f", $2/1048576}' /proc/meminfo)"
    av="$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo)"
    ok "内存：总 ${tot} GiB / 当前可用 ${av:-n/a} GiB"
    if [ "$WITH_OSTRICH" = "1" ] && awk "BEGIN{exit !($tot < 27)}"; then
      warn "Ostrich 需要把 WSL2 内存上限提到 28 GB（本机当前 ${tot} GiB）。"
      info "WSL2 默认只拿物理内存的一半。在 Windows 侧建 C:\\\\Users\\\\<你>\\\\.wslconfig："
      info "    [wsl2]"
      info "    memory=28GB"
      info "    swap=16GB"
      info "改完执行 wsl --shutdown 再进入。内存不够就改用 --no-ostrich（只有 5 格）。"
    fi
  else
    warn "读不到 /proc/meminfo，跳过内存检查"
  fi

  local par; par="$(dirname -- "$QROOT")"
  if command -v df >/dev/null 2>&1; then
    ok "磁盘（$par）：$(df -h "$par" 2>/dev/null | awk 'NR==2{print $4" 可用 / "$2" 总"}')"
    info "4 套 venv + 3 个 submodule 约需 30~40 GB"
  fi

  local py
  py="$(command -v python3 || command -v python || true)"
  if [ -n "$py" ]; then ok "python：$py（$("$py" -V 2>&1)）"; else warn "PATH 里没有 python3（出图需要 numpy + matplotlib）"; fi

  [ -d "$QROOT/.git" ] && ok "Qflex 源码：已存在 $QROOT" || info "Qflex 源码：尚未克隆，setup 阶段会 clone 到 $QROOT"

  [ "$bad" = "0" ] && { say "检查通过"; return 0; }
  say "有阻塞项，见上面 [FAIL]"; return 1
}

# ---------------------------------------------------------------- 阶段 2：setup
stage_setup(){
  if [ "$DRYRUN" = "1" ]; then
    say "setup（干跑，不执行）"
    info "git clone --recurse-submodules https://github.com/LNSGroup/Qflex.git $QROOT"
    info "git -C $QROOT apply $HERE/patches/0001-train-expose-num-workers-and-save-policy-every.patch"
    info "git -C $QROOT apply $HERE/patches/0002-persistence-jax0438-compat.patch"
    info "QROOT=$QROOT bash $HERE/setup/setup_qflex.sh"
    info "QROOT=$QROOT bash $HERE/setup/setup_other_envs_v3.sh"
    info "QROOT=$QROOT bash $HERE/setup/fix_humanoidbench.sh"
    return 0
  fi

  say "1/3 取官方代码"
  if [ -d "$QROOT/.git" ]; then
    ok "已存在 $QROOT，只更新 submodule"
    git -C "$QROOT" submodule update --init --recursive || { no "submodule 更新失败"; return 1; }
  else
    mkdir -p "$(dirname -- "$QROOT")"
    git clone --recurse-submodules https://github.com/LNSGroup/Qflex.git "$QROOT" || { no "clone 失败（检查网络/GitHub 可达性）"; return 1; }
  fi

  say "2/3 打补丁（幂等：已打过就跳过）"
  local p
  for p in \
    "$HERE/patches/0001-train-expose-num-workers-and-save-policy-every.patch" \
    "$HERE/patches/0002-persistence-jax0438-compat.patch" ; do
    if git -C "$QROOT" apply --reverse --check "$p" >/dev/null 2>&1; then
      ok "已打过：$(basename "$p")"
    elif git -C "$QROOT" apply --check "$p" >/dev/null 2>&1; then
      git -C "$QROOT" apply "$p" && ok "已应用：$(basename "$p")" || { no "应用失败：$(basename "$p")"; return 1; }
    else
      no "既没打过也打不上：$(basename "$p")（源码版本可能已变）"
      info "先看： git -C \"$QROOT\" apply --check -v \"$p\""
      return 1
    fi
  done

  say "3/3 建 4 套 venv（幂等，可重跑；约 40~60 min）"
  QROOT="$QROOT" bash "$HERE/setup/setup_qflex.sh"         || { no "setup_qflex.sh 失败"; return 1; }
  QROOT="$QROOT" bash "$HERE/setup/setup_other_envs_v3.sh" || { no "setup_other_envs_v3.sh 失败"; return 1; }
  QROOT="$QROOT" bash "$HERE/setup/fix_humanoidbench.sh"   || { no "fix_humanoidbench.sh 失败"; return 1; }
  ok "setup 完成"
}

# ---------------------------------------------------------------- 阶段 3：verify
stage_verify(){
  say "校验 7 个环境"
  if [ "$DRYRUN" = "1" ]; then info "[dry] QROOT=$QROOT bash $HERE/setup/verify_envs.sh"; return 0; fi
  QROOT="$QROOT" bash "$HERE/setup/verify_envs.sh"
}

# ---------------------------------------------------------------- 阶段 4：train
stage_train(){
  say "跑批：$(( $(printf '%s\n' $SEEDS | wc -l) )) 个种子 × $NENV 环境 × 6 算法"
  info "已完成的 run 会自动跳过；中断后重跑同一条命令即可续上"
  # NO_TMUX=1：本脚本已保证自己在一个可挂断的会话里，不要再让下层另外开一个
  NO_TMUX=1 ROOT="$QROOT" SEEDS="$SEEDS" WITH_OSTRICH="$WITH_OSTRICH" \
    WATCH_MB="$WATCH_MB" ALGS="$ALGS" ONLY="$ONLY" SKIP="$SKIP" LOGDIR="$LOGDIR" \
    bash "$HERE/run/run_seeds_chain.sh"
}

# ---------------------------------------------------------------- 阶段 5：plot
pick_python(){
  local c
  for c in "$QROOT/.venv/relax_myosuite/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
    [ -n "$c" ] && [ -x "$c" ] || continue
    "$c" -c 'import matplotlib, numpy' >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  return 1
}

stage_plot(){
  say "出图"
  if [ "$DRYRUN" = "1" ]; then
    info "[dry] QFLEX_LOGS=$QROOT/logs QFLEX_OUT=$OUT python $HERE/plot/export_6grid_3seed.py"
    return 0
  fi
  local py
  if ! py="$(pick_python)"; then
    no "找不到同时有 numpy + matplotlib 的 python"
    info "装一个： python3 -m pip install matplotlib numpy"
    info "或指定解释器： PYTHON=/path/to/python bash \"$SELF\" plot"
    return 1
  fi
  ok "解释器：$py"
  mkdir -p "$OUT"
  QFLEX_LOGS="$QROOT/logs" QFLEX_OUT="$OUT" "${PYTHON:-$py}" "$HERE/plot/export_6grid_3seed.py"
  local rc=$?
  [ $rc -eq 0 ] && ok "图已输出到 $OUT" || no "出图失败（rc=$rc）；若 log.csv 缺失，核对 QFLEX_LOGS=$QROOT/logs"
  return $rc
}

# ---------------------------------------------------------------- 主流程
print_plan
if is_long "$CMD" && [ "$ASSUME_YES" != "1" ] && [ "$DRYRUN" != "1" ] && [ "${QFB_REEXEC:-0}" != "1" ]; then
  plan_cost
  echo
  printf '回车开始 / Ctrl-C 取消： '
  read -r _ || true
  echo
fi

# ---------------------------------------------------------------- tmux 重入
# 确认之后才进会话：这样上面那句提示是在你自己的终端里等的（detached 会话里 read 读不到输入）
if is_long "$CMD" && [ -z "${TMUX:-}" ] && [ "$NO_TMUX" != "1" ] && [ "$DRYRUN" != "1" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    no "没有 tmux，长任务会被终端断开带走；先 sudo apt-get install -y tmux，或加 --no-tmux 自行承担"
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    no "tmux 会话已存在：$SESSION（先 tmux kill-session -t $SESSION，或改 SESSION=）"
    exit 1
  fi
  MLOG="$LOGDIR/main_$(date +%m%d_%H%M%S).log"
  mkdir -p "$LOGDIR"
  tmux new-session -d -s "$SESSION" "QFB_REEXEC=1 bash '$SELF' $CMD 2>&1 | tee -a '$MLOG'"
  ok "已放进 tmux 会话 $SESSION"
  info "日志  ：$MLOG"
  info "attach：tmux attach -t $SESSION    （Ctrl-b 再按 d 退出，不影响运行）"
  info "中止  ：tmux kill-session -t $SESSION"
  info "续跑  ：重跑同一条命令（已完成的 run 会跳过）"
  exit 0
fi

T0=$(date +%s)
RC=0
case "$CMD" in
  check)  stage_check  || RC=1 ;;
  setup)  stage_setup  || RC=1 ;;
  verify) stage_verify || RC=1 ;;
  train)  stage_train  || RC=1 ;;
  plot)   stage_plot   || RC=1 ;;
  all)
    stage_check  || { no "前置检查未通过，已停止"; RC=1; }
    if [ $RC -eq 0 ]; then stage_setup || RC=1; fi
    if [ $RC -eq 0 ]; then stage_verify || { no "环境校验未通过，已停止（不要带着失败往下跑）"; RC=1; }; fi
    if [ $RC -eq 0 ]; then stage_train || RC=1; fi
    # 出图不该掩盖训练结果，rc 单独记
    if [ $RC -eq 0 ]; then stage_plot || warn "出图未成功（不影响已完成的训练）"; fi
    ;;
esac

DT=$(( $(date +%s) - T0 ))
say "结束：$CMD  用时 $((DT/3600))h$(((DT%3600)/60))m  rc=$RC"
if [ "$CMD" = "all" ] || [ "$CMD" = "train" ]; then
  info "看进度：bash \"$HERE/run/qflex_batch_progress.sh\""
fi
exit $RC
