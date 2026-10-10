#!/usr/bin/env bash
# ============================================================================
# run_seeds_chain.sh — 串行补种子驱动器（把多个 SEED 串成一条无人值守的链）
#
# 为什么需要它：驱动器脚本（默认 = 本目录的 run_envs_formal_v5.sh）一次只接受一个
#   SEED，而它自己有这个逻辑 ——「不在 tmux 内 → 建会话后立刻 exit 0」。所以**直接在
#   外面写 for 循环会让 4 个 seed 同时开跑**（每个都只建会话就返回），抢同一块 GPU
#   与内存 ⇒ 必炸。因此必须：本脚本常驻 tmux，循环里用 NO_TMUX=1 让驱动器在
#   **前台**跑完再进下一个。
#
# 断点续跑：驱动器用 `.formal_done` + `.formal_total` 判据，已完成的 run 自动跳过
#   ⇒ 链中途断了（断电 / 手动 kill / WSL 重启）只要重跑同一条命令即可续上，
#     已烧掉的时间不会重烧。默认总是从 SEEDS 的第一个开始，前面几个只会「全跳过」
#     （30 行 skip，几秒钟），这是刻意设计 —— 不需要状态文件。
#
# 失败即停：某个 seed 出现失败 run 时**停止整条链**。理由：带着失败往下跑，
#   后面的 seed 也会缺同一格，而补种子的验收要求是「5 格 × N seeds 全齐」，
#   缺一格等于这条曲线白跑。修好后重跑同一条命令即可（已完成的会被跳过）。
#
# 用法：
#   SEEDS="100 101 102" bash ~/run_seeds_chain.sh   # 跑论文 Figure 2(b) 的 3 个种子（5 环境 × 6 算法）
#   SEEDS="100 101 102" WITH_OSTRICH=1 bash ~/run_seeds_chain.sh
#        ★ 跑满 6 环境（= 6×6×3 = 108 run）必须加 WITH_OSTRICH=1；
#          不加就只有 5 环境（90 run），出图只有 5 格。WATCH_MB 由驱动器脚本自动降为 1200，不用手加。
#   bash ~/run_seeds_chain.sh                 # 默认 SEEDS="101 102 103 104"（补种子模板）
#   SEEDS="101" bash ~/run_seeds_chain.sh     # 只补一个 seed（MyoLeg 已完成会自动跳）
#   SESSION=qflex_chain NO_TMUX=1 bash ~/run_seeds_chain.sh   # 已经在 tmux 里时
#   DRYRUN=1 bash ~/run_seeds_chain.sh        # 干跑：让驱动器只打印命令与计划、不开会话、不占 GPU
#
# 看进度：bash ~/qflex_batch_progress.sh   ｜   tail -f <链日志路径>
#   attach：tmux attach -t qflex_chain     （Ctrl-b 再按 d 退出，不影响运行）
#   中止  ：tmux kill-session -t qflex_chain
#           pkill -9 -f 'scripts/train.py'   # 训练是 setsid 出来的独立进程组，须补这一刀
# ============================================================================
set -uo pipefail

SEEDS="${SEEDS:-101 102 103 104}"
SESSION="${SESSION:-qflex_chain}"
# 默认调用**本目录**下的 run_envs_formal_v5.sh
# （原先的默认值指向家目录里的另一个驱动器脚本，本仓库不含它；改成"本目录"是为了不依赖外部文件）
VF="${VF:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)/run_envs_formal_v5.sh}"
ROOT="${ROOT:-$HOME/work/Qflex}"
LOGDIR="${LOGDIR:-$HOME/qflex_logs}"
mkdir -p "$LOGDIR"

# ★ 透传 WITH_OSTRICH / WATCH_MB：tmux 新会话**不保证**继承当前 shell 的环境变量
#   （tmux 默认只更新 update-environment 里列的那几个），所以既要 export，
#   又要显式写进 tmux 命令串、并在每个 seed 的子进程前缀里再给一遍。
#   ⇒ 以前这里没透传 ⇒ 即使外面写了 WITH_OSTRICH=1，链里也只会跑 5 个环境（README 的坑）。
WITH_OSTRICH="${WITH_OSTRICH:-0}"
export WITH_OSTRICH
WATCH_MB="${WATCH_MB:-}"
export WATCH_MB
PASS_ENV="WITH_OSTRICH='$WITH_OSTRICH' WATCH_MB='$WATCH_MB'"
if [ "$WITH_OSTRICH" = "1" ]; then NENV_PER_SEED=6; else NENV_PER_SEED=5; fi
NPER=$((NENV_PER_SEED * 6))

STAMP="$(date +%m%d_%H%M%S)"
WLOG="${WLOG:-$LOGDIR/seedchain_${STAMP}.log}"
SELF="$(readlink -f "$0" 2>/dev/null || echo "$0")"

say(){  printf '\033[1;36m%s\033[0m\n' "$1"; }
ok(){   printf '\033[1;32m[OK]\033[0m   %s\n' "$1"; }
no(){   printf '\033[1;31m[FAIL]\033[0m %s\n' "$1"; }
warn(){ printf '\033[1;33m[WARN]\033[0m %s\n' "$1"; }

mem_mb(){ awk -v k="$1" '$1==k":"{print int($2/1024)}' /proc/meminfo; }

# ---------- 前置检查（都在建 tmux 之前，失败要立刻可见）----------
[ -f "$VF" ]   || { no "找不到驱动器脚本（VF）：$VF"; exit 1; }
[ -d "$ROOT" ] || { no "找不到训练仓库：$ROOT"; exit 1; }

if pgrep -f 'scripts/train\.py' >/dev/null 2>&1; then
  no "已有训练进程在跑（scripts/train.py）—— 先等它收工或 kill，避免两批抢同一块 GPU"
  exit 1
fi

# ---------- 第 1 层：不在 tmux 内 → 自建会话后立刻返回 ----------
# （DRYRUN=1 时跳过这一层：只干跑，不开会话）
if [ -z "${TMUX:-}" ] && [ "${NO_TMUX:-0}" != "1" ] && [ "${DRYRUN:-0}" != "1" ]; then
  if ! command -v tmux >/dev/null 2>&1; then no "没有 tmux，请先：sudo apt-get install -y tmux"; exit 1; fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    no "tmux 会话已存在：$SESSION（先 kill 或换 SESSION=）"; exit 1
  fi
  tmux new-session -d -s "$SESSION" "WLOG='$WLOG' SEEDS='$SEEDS' VF='$VF' ROOT='$ROOT' $PASS_ENV NO_TMUX=1 bash '$SELF'"
  ok "已在 tmux 会话 $SESSION 里启动链式补种子"
  echo "     SEEDS=[$SEEDS]   WITH_OSTRICH=$WITH_OSTRICH"
  echo "     每个 seed = $NENV_PER_SEED 环境 × 6 算法 = $NPER run（已完成的自动跳过）"
  echo "     链日志  ：$WLOG"
  echo "     各seed日志：$LOGDIR/seedchain_seed<SEED>_*.log"
  echo "     汇总表  ：$LOGDIR/formal_summary_<stamp>.csv"
  echo
  echo "  看进度 : bash ~/qflex_batch_progress.sh"
  echo "  attach : tmux attach -t $SESSION     （Ctrl-b 再按 d 退出，不影响运行）"
  echo "  中止   : tmux kill-session -t $SESSION"
  echo "           pkill -9 -f 'scripts/train.py'    # 训练走 setsid 独立进程组，须补这一刀"
  echo "  续跑   : 断了以后重跑同一条命令即可（已完成的 run 自动跳过）"
  exit 0
fi

# ================= 以下在 tmux 会话内执行 =================
T_CHAIN0=$(date +%s)
{
  echo "===== SEED CHAIN START $(date '+%F %T') ====="
  echo "seeds=[$SEEDS]  driver=$VF  root=$ROOT  WITH_OSTRICH=$WITH_OSTRICH"
  echo "mem(before): total=$(mem_mb MemTotal)MB avail=$(mem_mb MemAvailable)MB swap=$(mem_mb SwapTotal)MB"
  if [ "$WITH_OSTRICH" = "1" ]; then
    echo "注意：一个完整 seed（6 环境 × 6 算法 = $NPER run）≈ 101 h（5 格 ≈73 h + Ostrich ≈28 h）"
    echo "      Ostrich 单 run 峰值 ≈24.5 GB，需 WSL2 上限 ≥28 GB；驱动器脚本会自动把看门狗水位降到 1200，无需手加"
  else
    echo "注意：一个完整 seed（5 环境 × 6 算法 = $NPER run）≈ 73 h（dacer 占 60%）"
    echo "      ⚠ 不含 Ostrich ⇒ 出图只有 5 格。要跑满 6 格请加 WITH_OSTRICH=1"
  fi
} | tee -a "$WLOG"

NSEED=0; NFAIL=0; NSTOP=0
for S in $SEEDS; do
  NSEED=$((NSEED+1))
  SLOG="$LOGDIR/seedchain_seed${S}_$(date +%m%d_%H%M%S).log"
  {
    echo ""
    echo "########## SEED $S 开始 $(date '+%F %T')   本 seed 日志：$SLOG ##########"
  } | tee -a "$WLOG"

  T0=$(date +%s)
  NO_TMUX=1 SEED="$S" SESSION="qflex_s${S}" DRYRUN="${DRYRUN:-0}" \
    WITH_OSTRICH="$WITH_OSTRICH" WATCH_MB="$WATCH_MB" bash "$VF" 2>&1 \
    | tee -a "$SLOG" | tee -a "$WLOG"
  rc="${PIPESTATUS[0]}"
  DT=$(( $(date +%s) - T0 ))
  FAILLINE="$(grep -m1 '^失败：' "$SLOG" 2>/dev/null || true)"

  {
    echo "########## SEED $S 结束 $(date '+%F %T')   用时 $((DT/3600))h$(((DT%3600)/60))m   rc=$rc ##########"
    echo "           ${FAILLINE:-（未拿到结果自检行）}"
  } | tee -a "$WLOG"

  if [ "${DRYRUN:-0}" = "1" ]; then
    ok "SEED $S 干跑完成（DRYRUN=1，未更新任何产物）"
    continue
  fi

  if [ "$rc" != "0" ]; then
    NSTOP=1
    warn "SEED $S 非零退出（rc=$rc）⇒ 停止整条链"
    echo "[STOP] 先看 $SLOG 的尾部定位原因；修好后重跑同一条命令即可（已完成的会跳过）" | tee -a "$WLOG"
    break
  fi
  if [ -z "$FAILLINE" ]; then
    NSTOP=1
    warn "SEED $S 未拿到「失败：」自检行 ⇒ 停止整条链（多半是提前 FAIL 退出）"
    echo "[STOP] 先看 $SLOG 的尾部定位原因；修好后重跑同一条命令即可（已完成的会跳过）" | tee -a "$WLOG"
    break
  fi
  if [ "$FAILLINE" != "失败：（无）" ]; then
    NFAIL=$((NFAIL+1))
    NSTOP=1
    warn "SEED $S 有失败 run ⇒ 停止整条链（带着失败往下跑，后面 seed 会缺同一格）"
    echo "[STOP] 失败行：$FAILLINE" | tee -a "$WLOG"
    echo "[STOP] 看 $SLOG 里对应的 run 日志；修好后重跑同一条命令即可（已完成的会跳过）" | tee -a "$WLOG"
    break
  fi
  ok "SEED $S 完成（$(date '+%F %T')）"
done

T_CHAIN=$(( $(date +%s) - T_CHAIN0 ))
{
  echo ""
  echo "===== SEED CHAIN DONE $(date '+%F %T') ====="
  echo "处理 seed 数：$NSEED   被失败中断：$NSTOP"
  echo "总用时：$((T_CHAIN/3600))h$(((T_CHAIN%3600)/60))m"
  echo "各 seed 日志：$LOGDIR/seedchain_seed*.log"
  echo "汇总表：$LOGDIR/formal_summary_*.csv"
  echo "下一步：bash ~/qflex_batch_progress.sh"
} | tee -a "$WLOG"
