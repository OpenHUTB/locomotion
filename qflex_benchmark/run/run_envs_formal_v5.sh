#!/usr/bin/env bash
# ============================================================================
# run_envs_formal_v5.sh — 全 6 环境批量训练（官方口径，串行）
#
# 范围：MyoLeg-Walk / MyoHand-PenTwirl / H1-Run / SMPL-Jump / H1-Balance × 6 算法
#       + Ostrich-Run（**可选**，默认关闭，见下）
#
# ★ 关于文件名里的 "v5"：它是这批脚本在开发过程中的迭代版本号，本仓库只含最终版，
#   v2/v3/v4 都不在仓库内。下面保留的"修正记录"是当时的实测依据，仅作背景。
#
# ★ Ostrich-Run 在 TABLE 里是**可选行，默认不启用**。
#   启用方式：WITH_OSTRICH=1（或直接 ONLY=OstrichRun-v1）。
#   为什么默认关闭：Ostrich 会用掉 ~24.5 GB 内存、单算法约 3.7 h，一旦默认并入，
#   将来跑 `SEED=xxx bash run_envs_formal_v5.sh` 补种子时会**多出 6 个 run / ≈32 h**，
#   而补种子的范围是另一个决策 ⇒ 不做隐式扩大。
#   Ostrich 参数（README.md:191 官方命令）：
#       env OstrichRun-v1 / venv relax_ms / 80 env / total 1e7 / --record_video 不加
#       workers 用 2 —— 本机实测结论：官方默认（=nenv=80 进程）不可行；
#       w=5 在 26,865 MiB 被内存逼停（rc=137），w=2 跑满 1e6 步成功、峰值 24,470 MiB、
#       757 步/秒 ⇒ 官方 1e7 步 ≈ 3.67 h/算法。w=4 未测（模型预测仅超自设水位 275 MiB）。
#   ★ 看门狗水位**不用手动加**：只要计划里含 Ostrich 且你没显式给 WATCH_MB，
#     脚本会自动把它设成 1200（见下方 "Ostrich 自动降看门狗水位" 段）。
#
# ★★ 想要"跑满 6 环境"（论文 Figure 2(b) 的 6 格 / 6×6×3 = 108 run）：
#      WITH_OSTRICH=1 bash run_envs_formal_v5.sh
#    不带 WITH_OSTRICH 只跑 5 环境 = 90 run（README 第 4 步的示例命令已带该开关）。
#    推荐入口是 run_seeds_chain.sh（它会把 WITH_OSTRICH / WATCH_MB 原样透传到每个 seed）。
#
# ★ TABLE 里包含 MyoLeg-Walk（而不是把它留给别的脚本单独跑）。
#   为什么必须并入：补种子要求**覆盖全部 5 格**，否则 Figure 2(b)/Fig.4 里只有
#   4 格有误差带、MyoLeg 仍是单条线，5 格不齐。
#   MyoLeg 官方参数（Qflex README 的官方命令）与其余四格同口径：
#       80 env / 2e7 步 / workers 20 / save_policy_every 1e6 (= total/20)
#   ⇒ 可并入同一张 TABLE，出图时天然同口径。
#
# ★ 看门狗必须**真杀**，不能只告警：
#   2026-09-26 17:29 那次训练就是被内核 OOM 连 tmux 一起带走的，起因是旧脚本的
#   "看门狗"只 printf 记录、**从不 kill**。本脚本按真杀机制重写：
#   2s 采样 + setsid --wait 取真 PGID + 超线 kill -9 -PGID（整组，含 worker）。
#
# 官方口径（逐条核过 README.md:168-193）：
#   h1-run-v0                70 env  5e6 步
#   h1-balance_simple-v0     70 env  2e7 步
#   SMPLHumanoidJump-v1      80 env  5e6 步
#   myoHandPenTwirlRandom-v0 80 env  5e6 步
#   * --num_workers 必须显式给（不传 ⇒ process3.py:22-24 兜底 = nenv 个 worker ⇒ h1 必炸）
#   * --save_policy_every = total/20（= 官方默认值，显式给以免被默认逻辑切错）
#   * 不加 --record_video（与已完成的 MyoLeg 一格保持一致）
#
# 用法：
#   DRYRUN=1 bash ~/run_envs_formal_v5.sh    # 只打印命令与计划，不执行（先看这个）
#   DRYRUN=1 WITH_OSTRICH=1 bash ~/run_envs_formal_v5.sh   # 看"跑满 6 环境"的计划
#   bash ~/run_envs_formal_v5.sh             # SEED=100（已跑过 ⇒ 30 个 run 全部"跳过"）
#
#   ★★ 补种子（本次要做的事）：只换 SEED，断点续跑会自动只跑新 seed
#   SEED=101 SESSION=qflex_seed101 bash ~/run_envs_formal_v5.sh
#        ≈ 73.3 h ≈ 3.1 天（5 环境 × 6 算法 = 30 run；dacer 单算法占 56%）
#   SEED=101 ONLY=myoLegWalk-v0 SESSION=qflex_s101_myoleg bash ~/run_envs_formal_v5.sh
#        ≈ 26.3 h ≈ 1.1 天（只补验收那一格，最快见效）
#
#   为什么换 SEED 是安全的：断点续跑 glob = "$ENV/${ALG}_*_s${SEED}_${SUFFIX}"，
#   含 seed 与 suffix；SEED=101 ⇒ SUFFIX 默认 s101 ⇒ glob 成 "_s101_s101"，
#   与已有的 "_s100_s100" 零重叠 ⇒ 不会误判"已存在"，也不会覆盖旧数据。
#
#   （PREFLIGHT=1 只对"从没跑过的环境"有意义；本次 5 格都跑过，用不上。）
#
#   ONLY="myoLegWalk-v0" bash ~/run_envs_formal_v5.sh      # 只跑某些环境
#   ALGS="qflex sac" bash ~/run_envs_formal_v5.sh          # 只跑部分算法
#   WORKERS_OVERRIDE="h1-run-v0=10" bash ~/run_envs_formal_v5.sh   # ★ 必须 KEY=value
#
# ★ 断点续跑与结果自检的设计要点（依据 = 2026-09-26 22:11 实测到的两个静默 bug）：
#   1. 断点续跑 glob 必须带 "${ALG}_" 前缀 —— 少了它，每格跑完第一个算法（qflex）后，
#      同环境其余 5 个算法会全部命中 glob 被误判"已存在"而跳过 ⇒ 每格只出 1 个 run
#      就"成功"结束。
#   2. 完成判据 = run 目录里的 .formal_done 标记（本脚本成功跑完时自己写）+
#      .formal_total 与本次 total 一致；**不能只看 log.csv 行数**（被看门狗 kill 掉的
#      半截 run 也有 log.csv 行，会被当成"已完成"而永远补不上）。
#   3. 末尾加"结果自检"：每格应产出 6 个 run，缺哪个算法直接 [WARN] 打出来。
#   4. 预检模式的 save_policy_every 改成 total/2：单次评估约 16 s（h1-run 实测），
#      5e4 步配 2500 的间隔要评估 19 次 = 5 分钟，占预检八成时间。
# 看进度：bash ~/qflex_batch_progress.sh  ｜  tail -f ~/qflex_logs/formal_<stamp>.log
# 中止  ：tmux kill-session -t <你启动时用的 SESSION 名>
#         ★ 再补一刀（训练进程是 setsid 出来的独立进程组，kill-session 打不到它）：
#           pkill -9 -f 'scripts/train.py'
# ============================================================================
set -uo pipefail

ALGS="${ALGS:-qflex sac crossq qsm sdac dacer}"
SEED="${SEED:-100}"
SESSION="${SESSION:-qflex_formal}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
GAP="${GAP:-20}"
NO_TMUX="${NO_TMUX:-0}"
PREFLIGHT="${PREFLIGHT:-0}"
DRYRUN="${DRYRUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
NENV_OVERRIDE="${NENV_OVERRIDE:-}"
WORKERS_OVERRIDE="${WORKERS_OVERRIDE:-}"
WATCH_MB_RAW="${WATCH_MB:-}"       # 用户是否显式给过（空 ⇒ 允许后面按环境自动设定，见下方 Ostrich 段）
WATCH_MB="${WATCH_MB:-3000}"
SWAP_MIN_MB="${SWAP_MIN_MB:-2048}"
RUN_TIMEOUT="${RUN_TIMEOUT:-129600}"     # 单 run 硬超时（秒），默认 36 h
#   ★ 为什么从 24 h 抬到 36 h：本批最慢的组合是 h1-balance × dacer × 2e7 步，
#     按 myoLeg 实测（dacer 2e7 = 14.07 h）再按 h1-balance/qflex 比 myoLeg 快 1.19 倍
#     折算 ≈ 11.8~13.4 h；24 h 只有约 1.8 倍余量，一旦跨环境变慢就可能在第 24 小时被砍，
#     而 v3 的完成标记会让它下次重启时**从头重跑**，等于白烧一整天 GPU。
ROOT="${ROOT:-$HOME/work/Qflex}"

if [ "$PREFLIGHT" = "1" ]; then
  TOT_OVERRIDE=50000
  SUFFIX="${SUFFIX:-pre}"
else
  TOT_OVERRIDE=0
  SUFFIX="${SUFFIX:-s${SEED}}"
fi

# 执行顺序 = 表格顺序（按每格 qflex 工期升序：先拿最便宜的完整结果）
# env | venv | nenv(官方) | total(官方) | workers
TABLE='
myoHandPenTwirlRandom-v0|relax_myosuite|80|5000000|20
h1-run-v0|relax_humanoidbench|70|5000000|14
SMPLHumanoidJump-v1|relax_smplhumanoid|80|5000000|20
myoLegWalk-v0|relax_myosuite|80|20000000|20
h1-balance_simple-v0|relax_humanoidbench|70|20000000|14
'
# ★ Ostrich-Run 单独一行、**默认不并入**（见文件头）。列含义：env|venv|nenv|total|workers
if [ "${WITH_OSTRICH:-0}" = "1" ]; then
  TABLE="${TABLE}OstrichRun-v1|relax_ms|80|10000000|2
"
fi

LOGDIR="$HOME/qflex_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%m%d_%H%M%S)"
MAINLOG="$LOGDIR/formal_${STAMP}.log"
SUM="$LOGDIR/formal_summary_${STAMP}.csv"

C_TI=$'\033[1;36m'; C_GO=$'\033[1;32m'; C_NO=$'\033[1;31m'; C_0=$'\033[0m'
say(){  printf '%s%s%s\n' "$C_TI" "$1" "$C_0"; }
ok(){   printf '%s[OK]%s   %s\n' "$C_GO" "$C_0" "$1"; }
no(){   printf '%s[FAIL]%s %s\n' "$C_NO" "$C_0" "$1"; }

mem_avail()  { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
swap_free()  { awk '/SwapFree/{print int($2/1024)}'     /proc/meminfo; }
mem_total()  { awk '/MemTotal/{print int($2/1024)}'     /proc/meminfo; }
swap_total() { awk '/SwapTotal/{print int($2/1024)}'    /proc/meminfo; }

lookup() {  # $1 = name=value 列表, $2 = key
  local kv
  for kv in $1; do
    case "$kv" in "$2="*) printf '%s' "${kv#*=}"; return 0;; esac
  done
  return 1
}

# ---------- 展开计划（tmux 内外都要用，故先算） ----------
PLAN=""   # 每行：ENV|VENV|NENV|TOT|WORKERS|SAVE
while IFS='|' read -r ENV VENV NENV TOT WORKERS; do
  [ -z "${ENV:-}" ] && continue
  case " $SKIP " in *" $ENV "*) continue;; esac
  if [ -n "$ONLY" ]; then
    case " $ONLY " in *" $ENV "*) : ;; *) continue;; esac
  fi
  if OV="$(lookup "$NENV_OVERRIDE" "$ENV")"; then NENV="$OV"; fi
  if [ "$TOT_OVERRIDE" != "0" ]; then TOT="$TOT_OVERRIDE"; fi
  if OV="$(lookup "$WORKERS_OVERRIDE" "$ENV")"; then
    if [ "$OV" -ge 1 ] && [ $((NENV % OV)) -eq 0 ]; then WORKERS="$OV"
    else echo "[WARN] $ENV：WORKERS_OVERRIDE=$OV 不能整除 NENV=$NENV，忽略" >&2; fi
  fi
  if [ $((NENV % WORKERS)) -ne 0 ]; then
    no "$ENV：workers=$WORKERS 不整除 nenv=$NENV（process3.py:26 会 assert 失败）" >&2
    exit 1
  fi
  SAVE=$(( TOT / 20 ))
  # ★ 预检专用（v3 改动 4）：把评估间隔放大到 total/2（= 只评估 2 次）。
  #   实测依据（2026-09-26 22:17，h1-run/qflex 预检）：单次评估成本只跟**环境**有关、
  #   跟 total_step 无关 —— h1-run 一次评估 ≈ 16 s（log.csv 每行间隔 15~30 s）。
  #   该轮 5e4 步里训练只占 48 s，19 次评估却占了 316 s（≈5.3 min），是整轮八成开销。
  #   预检的目的是"证明 6 个算法在 4 个新环境上都起得来 + 评估链路能落 log.csv"，
  #   2 次评估足够，可把预检从 ~2 h 压到 ~30 min。★ 正式跑不受影响（仍 total/20）。
  if [ "$PREFLIGHT" = "1" ]; then SAVE=$(( TOT / 2 )); fi
  [ "$SAVE" -lt 1 ] && SAVE=1
  PLAN="${PLAN}${ENV}|${VENV}|${NENV}|${TOT}|${WORKERS}|${SAVE}
"
done <<< "$(printf '%s' "$TABLE" | sed '/^$/d')"

N_RUNS=0
for a in $ALGS; do N_RUNS=$((N_RUNS + 1)); done
N_ENVS="$(printf '%s' "$PLAN" | grep -c . || true)"
if [ "${N_ENVS:-0}" -eq 0 ]; then echo "[FAIL] 计划为空（检查 ONLY/SKIP）" >&2; exit 1; fi

# ---------- ★ Ostrich 自动降看门狗水位（消除"忘了加 WATCH_MB=1200 就必被误杀"的陷阱）----------
# 机制：Ostrich 单 run 峰值 used ≈24.5 GB / WSL MemTotal ≈28 GB ⇒ 训练中 avail 会掉到 ~2 GB，
#   默认水位 3000 MB 会在**每一次** Ostrich run 上触发看门狗 kill -9（整组）。
# 规则：只要计划里含 Ostrich，且用户**没有显式给 WATCH_MB**，就自动设成 1200。
#   显式给过则以用户为准（不覆盖），但会在启动横幅里回显出来便于自查。
HAVE_OSTRICH="0"; printf '%s' "$PLAN" | grep -q '^OstrichRun-v1|' && HAVE_OSTRICH="1"
AUTO_WATCH="0"
if [ "$HAVE_OSTRICH" = "1" ] && [ -z "$WATCH_MB_RAW" ]; then
  WATCH_MB=1200; AUTO_WATCH="1"
fi

# ---------- DRYRUN ----------
if [ "$DRYRUN" = "1" ]; then
  say "DRYRUN：将执行 $N_ENVS 个环境 × $N_RUNS 个算法 = $((N_ENVS * N_RUNS)) 个 run"
  echo "ALGS=[$ALGS]  SEED=$SEED  SUFFIX=$SUFFIX  PREFLIGHT=$PREFLIGHT"
  echo
  MISS=0
  while IFS='|' read -r ENV VENV NENV TOT WORKERS SAVE; do
    [ -z "${ENV:-}" ] && continue
    # ★ 必须在这里也校验：否则 venv 名打错（曾经 relax_splhumanoid）DRYRUN 看不出来，
    #   要等真正启动才 [FAIL] 退出。注意只能在 WSL 内用 -x 判，Windows UNC 侧符号链接读不了。
    if [ -x "$ROOT/.venv/$VENV/bin/python" ]; then VP="venv OK"
    else VP="venv ** MISSING **"; MISS=$((MISS + 1)); fi
    printf '%-26s %-24s %s\n' "$ENV" "$VENV" "$VP"
  done <<< "$(printf '%s' "$PLAN" | sed '/^$/d')"
  echo
  while IFS='|' read -r ENV VENV NENV TOT WORKERS SAVE; do
    [ -z "${ENV:-}" ] && continue
    printf '%-26s %-22s nenv=%-4s workers=%-3s total=%-9s save_every=%-8s\n' \
      "$ENV" "$VENV" "$NENV" "$WORKERS" "$TOT" "$SAVE"
    for ALG in $ALGS; do
      echo "    $VENV/bin/python scripts/train.py --alg $ALG --env $ENV --seed $SEED \\"
      echo "        --total_step $TOT --num_vec_envs $NENV --num_workers $WORKERS \\"
      echo "        --save_policy_every $SAVE --suffix $SUFFIX"
    done
    echo
  done <<< "$(printf '%s' "$PLAN" | sed '/^$/d')"
  echo "预计总工期（5 格为**实测**值；Ostrich 一栏含外推，已注明）："
  echo "  myoHand 6.02 h ｜ h1-run 6.65 h ｜ SMPL 6.96 h ｜ myoLeg 26.32 h ｜ h1-balance 27.35 h"
  echo "  ⇒ 5 格合计 ≈ 73.3 h ≈ 3.1 天（dacer 单算法 ≈ 40.9 h，占 56%）"
  if [ "$HAVE_OSTRICH" = "1" ]; then
    echo "  ＋ OstrichRun-v1（1e7 步 @workers=2）：qflex **实测** 3.67 h/算法；"
    echo "     其余 5 算法速率未测，按 MyoLeg 的算法间耗时比外推 ⇒ 6 算法 ≈ 32 h ≈ 1.3 天（外推）。"
    if [ "$AUTO_WATCH" = "1" ]; then
      echo "     ⚠ Ostrich 峰值 used ≈24.5 GB ⇒ 看门狗水位**已自动下调**为 WATCH_MB=$WATCH_MB（默认 3000 会误杀）"
    else
      echo "     ⚠ 你显式指定了 WATCH_MB=$WATCH_MB —— Ostrich 需要 ≤1200，请自查（峰值 used ≈24.5 GB / WSL MemTotal ≈28 GB）"
    fi
  fi
  if [ "$MISS" -gt 0 ]; then
    no "有 $MISS 个 venv 路径不存在 —— 上面标 ** MISSING ** 的行是错的，先修 TABLE" >&2
    exit 1
  fi
  exit 0
fi

# ---------- 前置检查 ----------
if [ ! -d "$ROOT" ]; then echo "[FAIL] 找不到 $ROOT" >&2; exit 1; fi
if ! setsid --wait true >/dev/null 2>&1; then
  echo "[FAIL] 没有 setsid --wait —— 看门狗无法整组杀训练进程（需 util-linux 新于 2.24）" >&2
  exit 1
fi
while IFS='|' read -r ENV VENV NENV TOT WORKERS SAVE; do
  [ -z "${ENV:-}" ] && continue
  if [ ! -x "$ROOT/.venv/$VENV/bin/python" ]; then
    no "缺 venv：$ROOT/.venv/$VENV/bin/python" >&2; exit 1
  fi
done <<< "$(printf '%s' "$PLAN" | sed '/^$/d')"

# ---------- 第 1 层：不在 tmux 内 -> 自建会话后立刻返回 ----------
if [ -z "${TMUX:-}" ] && [ "$NO_TMUX" != "1" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[FAIL] 没有 tmux，请先：sudo apt-get install -y tmux" >&2; exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[WARN] tmux 会话已存在：$SESSION（先 kill 或换 SESSION=）" >&2; exit 1
  fi
  SELF="$(readlink -f "$0")"
  tmux new-session -d -s "$SESSION" "bash '$SELF'"
  echo "[OK] 已在 tmux 会话 $SESSION 里启动（跑完自动消失，不留死窗格）"
  echo "     ALGS=[$ALGS]  SEED=$SEED  SUFFIX=$SUFFIX  $N_ENVS 环境 × $((N_ENVS * N_RUNS)) run"
  echo "     主日志：$MAINLOG"
  echo "     汇总  ：$SUM"
  echo "     启动前内存：total=$(mem_total)MB avail=$(mem_avail)MB swap=$(swap_total)MB"
  echo
  echo "  看进度 : bash ~/qflex_batch_progress.sh"
  echo "  attach : tmux attach -t $SESSION     （Ctrl-b 再按 d 退出，不影响训练）"
  echo "  中止   : tmux kill-session -t $SESSION"
  exit 0
fi

# ================= 以下在 tmux 会话内执行 =================
cd "$ROOT" || exit 1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true'
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

{
  echo "===== FORMAL BATCH START $(date '+%F %T') ====="
  echo "algs=[$ALGS] seed=$SEED suffix=$SUFFIX preflight=$PREFLIGHT"
  echo "watchdog: WATCH_MB=${WATCH_MB}MB swap_min=${SWAP_MIN_MB}MB run_timeout=${RUN_TIMEOUT}s$([ "$AUTO_WATCH" = 1 ] && echo '  (WATCH_MB 因 Ostrich 自动下调)')"
  echo "plan:"
  printf '%s' "$PLAN" | sed '/^$/d' | sed 's/^/  /'
  echo "mem(before): total=$(mem_total)MB avail=$(mem_avail)MB swap=$(swap_total)MB"
} | tee -a "$MAINLOG"

# ---------- 内存看门狗（真杀版） ----------
PGIDFILE="$LOGDIR/.formal_pgid"
KILLFILE="$LOGDIR/.formal_kill"
PEAKFILE="$LOGDIR/.formal_peak"
STARTFILE="$LOGDIR/.formal_start"
: > "$PGIDFILE"; : > "$KILLFILE"; echo 0 > "$PEAKFILE"; echo 0 > "$STARTFILE"
(
  n=0
  while true; do
    sleep 2
    n=$((n + 1))
    av="$(mem_avail)"; sf="$(swap_free)"
    tot="$(mem_total)"; used=$((tot - av))
    pk="$(cat "$PEAKFILE" 2>/dev/null)"; pk=${pk:-0}
    if [ "$used" -gt "$pk" ]; then echo "$used" > "$PEAKFILE"; fi
    if [ $((n % 15)) -eq 0 ]; then
      printf '[mem %s] avail=%sMB swapfree=%sMB used=%sMB peak=%sMB\n' \
        "$(date '+%T')" "$av" "$sf" "$used" "$(cat "$PEAKFILE" 2>/dev/null)" >> "$MAINLOG"
    fi
    pg="$(cat "$PGIDFILE" 2>/dev/null)"
    [ -z "$pg" ] && continue
    st="$(cat "$STARTFILE" 2>/dev/null)"; st=${st:-0}
    why=""
    if [ "$av" -lt "$WATCH_MB" ]; then
      why="MEM avail=${av}MB<${WATCH_MB}MB"
    elif [ "$sf" -lt "$SWAP_MIN_MB" ]; then
      why="SWAP swapfree=${sf}MB<${SWAP_MIN_MB}MB"
    elif [ "$st" -gt 0 ] && [ "$RUN_TIMEOUT" -gt 0 ] && \
         [ $(( $(date +%s) - st )) -gt "$RUN_TIMEOUT" ]; then
      why="TIMEOUT >${RUN_TIMEOUT}s"
    fi
    if [ -n "$why" ]; then
      echo "$why" > "$KILLFILE"
      echo "     [WATCHDOG] $why  ->  kill -9 -$pg" >> "$MAINLOG"
      kill -9 -"$pg" 2>/dev/null
      : > "$PGIDFILE"
    fi
  done
) &
WATCH_PID=$!
trap 'kill $WATCH_PID 2>/dev/null; : > "$PGIDFILE" 2>/dev/null' EXIT

echo "env|venv|alg|nenv|workers|total|save_every|rc|wall_s|peak_mb|kill_reason|rate|rundir" > "$SUM"

DONE=""; FAILED=""; SKIPPED=""
IDX=0
while IFS='|' read -r ENV VENV NENV TOT WORKERS SAVE; do
  [ -z "${ENV:-}" ] && continue
  VENV_PY="$ROOT/.venv/$VENV/bin/python"
  say "[$ENV]  venv=$VENV  nenv=$NENV  workers=$WORKERS  total=$TOT  save_every=$SAVE"
  echo "     extra:（无）   buffer 满仓预估按 4*(2|S|+|A|)+5 计，见方案文档" | tee -a "$MAINLOG"

  for ALG in $ALGS; do
    IDX=$((IDX + 1))

    # 断点续跑：同名 run 已完成则跳过
    # ★★ glob **必须**带 "${ALG}_" 前缀：run 目录名是 <alg>_<date>_s<seed>_<suffix>，
    #    少了算法名的话，第一个算法跑完后同环境其余 5 个算法会全部命中 glob 被跳过
    #    （2026-09-26 22:11 实测踩到，v2 的 bug）。
    # ★ 判据 = 本脚本成功结束时写的 .formal_done + .formal_total 与本次 total 一致；
    #    只看 log.csv 行数会把被看门狗 kill 的半截 run 也当成"已完成"。
    EXIST=""
    for d in "$ROOT/logs/$ENV/${ALG}_"*_s${SEED}_${SUFFIX}; do
      [ -d "$d" ] || continue
      if [ -f "$d/.formal_done" ] && [ "$(cat "$d/.formal_total" 2>/dev/null)" = "$TOT" ]; then EXIST="$d"; fi
    done
    if [ -n "$EXIST" ] && [ "$SKIP_DONE" = "1" ]; then
      echo "--- [$IDX] $ALG  已存在 → 跳过（$EXIST）" | tee -a "$MAINLOG"
      echo "$ENV|$VENV|$ALG|$NENV|$WORKERS|$TOT|$SAVE|skip|-|-|-|-|$EXIST" >> "$SUM"
      SKIPPED="$SKIPPED ${ENV}/${ALG}"
      continue
    fi

    LOG="$LOGDIR/formal_${ENV}_${ALG}_s${SEED}_${STAMP}.log"
    echo "" | tee -a "$MAINLOG"
    echo "--- [$IDX] $ENV / $ALG  开始 $(date '+%F %T')  avail=$(mem_avail)MB" | tee -a "$MAINLOG"
    echo "    日志：$LOG" | tee -a "$MAINLOG"

    SECONDS=0
    : > "$KILLFILE"; echo 0 > "$PEAKFILE"; echo "$(date +%s)" > "$STARTFILE"
    # setsid --wait + bash -c 'echo $$ ...' 取**真正的**进程组长 pid（exec 不换 pid）
    setsid --wait bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PGIDFILE" \
      "$VENV_PY" -u scripts/train.py \
      --alg "$ALG" --env "$ENV" --seed "$SEED" \
      --total_step "$TOT" --num_vec_envs "$NENV" --suffix "$SUFFIX" \
      --num_workers "$WORKERS" --save_policy_every "$SAVE" \
      > "$LOG" 2>&1 &
    TRAIN_PID=$!
    wait "$TRAIN_PID"; RC=$?
    : > "$PGIDFILE"; echo 0 > "$STARTFILE"
    WALL=$SECONDS
    KILLED="$(cat "$KILLFILE" 2>/dev/null)"; KILLED="${KILLED//|/ }"
    PEAK="$(cat "$PEAKFILE" 2>/dev/null)"; PEAK=${PEAK:-0}
    [ -n "$KILLED" ] && RC=137
    RATE=$(awk -v t="$TOT" -v s="$WALL" 'BEGIN{ if (s>0) printf "%.0f", t/s; else printf "0" }')
    RUNDIR="$(ls -dt "$ROOT/logs/$ENV/${ALG}_"*_s${SEED}_${SUFFIX} 2>/dev/null | head -1)"
    echo "$ENV|$VENV|$ALG|$NENV|$WORKERS|$TOT|$SAVE|$RC|$WALL|$PEAK|$KILLED|$RATE|$RUNDIR" >> "$SUM"

    if [ "$RC" -eq 0 ]; then
      # ★ 完成标记：只有 rc=0 才写，断点续跑靠它判定（见上面 EXIST 的注释）
      if [ -n "$RUNDIR" ] && [ -d "$RUNDIR" ]; then
        printf '%s\n' "$TOT" > "$RUNDIR/.formal_total"
        touch "$RUNDIR/.formal_done"
      else
        echo "     [WARN] rc=0 但找不到 run 目录（$ENV/$ALG），未写完成标记：$RUNDIR" | tee -a "$MAINLOG"
      fi
      ok "$ENV / $ALG 完成  ${WALL}s（$(awk -v s="$WALL" 'BEGIN{printf "%.2f", s/3600}') h）  peak=${PEAK}MB"
      DONE="$DONE ${ENV}/${ALG}"
    else
      no "$ENV / $ALG 失败 rc=$RC  ${WALL}s  peak=${PEAK}MB  ${KILLED:+（看门狗：$KILLED）}"
      [ "$RC" -eq 137 ] && [ -z "$KILLED" ] && \
        echo "       rc=137 且非看门狗 ⇒ 疑似内核 OOM，查：grep -i oom /var/log/kern.log | tail -5" | tee -a "$MAINLOG"
      FAILED="$FAILED ${ENV}/${ALG}"
    fi
    sleep "$GAP"
  done
done <<< "$(printf '%s' "$PLAN" | sed '/^$/d')"

# ---------- 结果自检（防"静默跳过 / 根本没跑出来"这类 bug 无声通过）----------
echo "" | tee -a "$MAINLOG"
say "结果自检：每格应有 $N_RUNS 个算法产出 log.csv + .formal_done"
{
  while IFS='|' read -r ENV VENV NENV TOT WORKERS SAVE; do
    [ -z "${ENV:-}" ] && continue
    C=0; M=""
    for ALG in $ALGS; do
      d="$(ls -dt "$ROOT/logs/$ENV/${ALG}_"*_s${SEED}_${SUFFIX} 2>/dev/null | head -1)"
      if [ -n "$d" ] && [ -f "$d/.formal_done" ] && [ -f "$d/log.csv" ] && \
         [ "$(wc -l < "$d/log.csv")" -gt 1 ]; then
        C=$((C + 1))
      else
        M="$M $ALG"
      fi
    done
    printf '  %-26s %s/%s' "$ENV" "$C" "$N_RUNS"
    if [ "$C" -lt "$N_RUNS" ]; then
      printf '   [WARN] 未产出：%s\n' "${M:-（无）}"
    else
      printf '   OK\n'
    fi
  done <<< "$(printf '%s' "$PLAN" | sed '/^$/d')"
} | tee -a "$MAINLOG"

# ---------- 预检产物归位（避免被出图脚本当成正式 run 捞走）----------
if [ "$PREFLIGHT" = "1" ]; then
  PRE_DEST="$LOGDIR/pre_runs"
  mkdir -p "$PRE_DEST"
  MOVED=0
  for d in "$ROOT"/logs/*/*_s${SEED}_${SUFFIX}; do
    [ -d "$d" ] || continue
    e="$(basename "$(dirname "$d")")"
    mv "$d" "$PRE_DEST/${e}__$(basename "$d")" 2>/dev/null && MOVED=$((MOVED + 1))
  done
  echo "预检产物已归位：$MOVED 个 run -> $PRE_DEST（logs/ 下只留正式口径）" | tee -a "$MAINLOG"
fi

{
  echo ""
  echo "===== FORMAL BATCH DONE $(date '+%F %T') ====="
  echo "成功：${DONE:-（无）}"
  echo "跳过：${SKIPPED:-（无）}"
  echo "失败：${FAILED:-（无）}"
  echo "汇总：$SUM"
  echo "各 run 目录：$ROOT/logs/<env>/"
  echo "下一步：bash ~/qflex_batch_progress.sh"
} | tee -a "$MAINLOG"
