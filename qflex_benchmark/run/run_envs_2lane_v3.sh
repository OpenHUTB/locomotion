#!/usr/bin/env bash
# ============================================================================
# run_envs_2lane.sh — 4 格「同环境两路」补种子（每路 workers=5）
#
# 只覆盖**实测可两路**的 4 个环境（逐环境标定结论，见交付附件 附录 E §8）：
#     h1-run-v0        S=1.842   两路峰值 24,984 MB
#     h1-balance_simple-v0 S=1.773  两路峰值 25,040 MB
#     myoLegWalk-v0    S=1.638   两路峰值 20,562 MB
#     SMPLHumanoidJump-v1  S=1.576  两路峰值  6,443 MB
# **不含** myoHandPenTwirlRandom-v0（实测两路 rc=137，真超内存，见 §8.2）
# **不含** OstrichRun-v1（单路峰值 ≈24.5 GB ⇒ 两路 ≈49 GB，物理上不可能）
# 这两个请用 `run_envs_formal_v5.sh` 串行补（见文件末尾的收尾提示）。
#
# ---------------------------------------------------------------------------
# 为什么是「同环境两路」而不是「跨环境配对」
#   跨环境配对（在一路空转时塞别的环境的任务）能把利用率拉近 S≈1.77，
#   但要引入未标定的跨环境 GPU 争用 + 异构内存组合，收益上限只有约 7.5 h/seed
#   ⇒ 性价比不足，本脚本不做。代价是：dacer 那条 lane 是**唯一的长尾**，
#   另一条 lane 跑完 5 个轻算法后空转（dacer 占单环境时长 53~58%）。
#   本脚本用 LPT（按实测耗时降序贪心）把长尾固定给 dacer，使 total ≈ max(lane)。
#
# ---------------------------------------------------------------------------
# 内存安全
#   两路 w=5 的峰值已在标定里逐环境实测（上表）。但**算法维度未标**——
#   标定两路跑的是同一个算法（qflex），正式跑两路是**两个不同算法同时**，
#   峰值可能略高 ⇒ 看门狗水位取 WATCH_MB=800（默认值），超线即整组杀。
#   注意：单路 w=5 只需 ~13 GB，WATCH_MB=800 只在两路叠加时可能触发。
#
# 用法：
#   DRYRUN=1 bash ~/run_envs_2lane_v3.sh                   # 看计划 + LPT 分配（不跑）
#   SEEDS="101" bash ~/run_envs_2lane.sh                # 补 seed 101 的 4 格
#   SEEDS="101 102" bash ~/run_envs_2lane.sh            # 补到 N=3
#   ONLY="h1-balance_simple-v0" SEEDS="101" bash ~/run_envs_2lane.sh   # 只跑一格
#   ALGS="qflex dacer" SEEDS="101" bash ~/run_envs_2lane.sh            # 只跑部分算法
#   LANES=1 bash ~/run_envs_2lane.sh                    # 退回串行（等价 v5，用于对照）
#   W_LO=2 bash ~/run_envs_2lane.sh                     # 两路装不下时降档复测
#
# 看进度：tail -f ~/qflex_logs/2lane_<stamp>.log  ｜ bash ~/qflex_batch_progress.sh
# 中止  ：tmux kill-session -t qflex_2lane
#         ★ 再补一刀（训练是 setsid 独立进程组，kill-session 打不到）：
#           pkill -9 -f 'scripts/train.py'
# 续跑  ：断了重跑同一条命令即可（.formal_done 标记会跳过已完成的 run）
# ============================================================================
set -uo pipefail

ALGS="${ALGS:-qflex sac crossq qsm sdac dacer}"
SEEDS="${SEEDS:-101}"
LANES="${LANES:-2}"
W_LO="${W_LO:-5}"
SESSION="${SESSION:-qflex_2lane}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
GAP="${GAP:-20}"
NO_TMUX="${NO_TMUX:-0}"
DRYRUN="${DRYRUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
WATCH_MB="${WATCH_MB:-800}"
SWAP_MIN_MB="${SWAP_MIN_MB:-2048}"
RUN_TIMEOUT="${RUN_TIMEOUT:-129600}"     # 单 run 硬超时 36 h（两路时 dacer 约 18 h）
ROOT="${ROOT:-$HOME/work/Qflex}"
LOGDIR="${LOGDIR:-$HOME/qflex_logs}"
mkdir -p "$LOGDIR"

STAMP="$(date +%m%d_%H%M%S)"
MAINLOG="$LOGDIR/2lane_${STAMP}.log"
SUM="$LOGDIR/2lane_summary_${STAMP}.csv"

# env | venv | nenv(官方) | total(官方) | W_HI(官方单路基线，仅用于展示/对照)
TABLE2='
h1-run-v0|relax_humanoidbench|70|5000000|14
SMPLHumanoidJump-v1|relax_smplhumanoid|80|5000000|20
myoLegWalk-v0|relax_myosuite|80|20000000|20
h1-balance_simple-v0|relax_humanoidbench|70|20000000|14
'

# LPT 用的**实测耗时表**（seed 100 的 wall_s，W_HI 单路；来源 formal_summary_*.csv）
# 只用来决定「哪些算法放同一条 lane」，不影响训练参数。
COST2='
h1-run-v0|dacer|14112
h1-run-v0|sdac|2592
h1-run-v0|qflex|2736
h1-run-v0|qsm|1800
h1-run-v0|crossq|1440
h1-run-v0|sac|1224
SMPLHumanoidJump-v1|dacer|13248
SMPLHumanoidJump-v1|sdac|3312
SMPLHumanoidJump-v1|qflex|2880
SMPLHumanoidJump-v1|qsm|2628
SMPLHumanoidJump-v1|crossq|1620
SMPLHumanoidJump-v1|sac|1404
myoLegWalk-v0|dacer|49716
myoLegWalk-v0|sdac|13392
myoLegWalk-v0|qflex|10620
myoLegWalk-v0|qsm|8748
myoLegWalk-v0|crossq|5976
myoLegWalk-v0|sac|5004
h1-balance_simple-v0|dacer|57168
h1-balance_simple-v0|sdac|10944
h1-balance_simple-v0|qflex|11304
h1-balance_simple-v0|qsm|7380
h1-balance_simple-v0|crossq|5940
h1-balance_simple-v0|sac|5724
'

C_TI=$'\033[1;36m'; C_GO=$'\033[1;32m'; C_NO=$'\033[1;31m'; C_WN=$'\033[1;33m'; C_0=$'\033[0m'
say(){  printf '%s%s%s\n' "$C_TI" "$1" "$C_0"; }
ok(){   printf '%s[OK]%s   %s\n' "$C_GO" "$C_0" "$1"; }
no(){   printf '%s[FAIL]%s %s\n' "$C_NO" "$C_0" "$1"; }
warn(){ printf '%s[WARN]%s %s\n' "$C_WN" "$C_0" "$1"; }

mem_avail()  { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
swap_free()  { awk '/SwapFree/{print int($2/1024)}'     /proc/meminfo; }
mem_total()  { awk '/MemTotal/{print int($2/1024)}'     /proc/meminfo; }
swap_total() { awk '/SwapTotal/{print int($2/1024)}'    /proc/meminfo; }
h2m(){ awk -v s="$1" 'BEGIN{
  if (s>=3600)      printf "%dh%02dm", int(s/3600), int((s%3600)/60);
  else if (s>=60)   printf "%dm%02ds", int(s/60), int(s%60);
  else              printf "%ds", int(s);
}'; }

lookup(){  # $1=name=value 列表, $2=key
  local kv
  for kv in $1; do
    case "$kv" in "$2="*) printf '%s' "${kv#*=}"; return 0;; esac
  done
  return 1
}

cost_of(){  # $1=env $2=alg  → 秒（查不到给 3600）
  local e a c
  while IFS='|' read -r e a c; do
    [ -z "${e:-}" ] && continue
    if [ "$e" = "$1" ] && [ "$a" = "$2" ]; then printf '%s' "$c"; return 0; fi
  done <<< "$(printf '%s' "$COST2" | sed '/^$/d')"
  printf '3600'
}

# ── 两路折算系数 k（**实测统一常数**，见 附录 E §11）──────────────────────
# COST2 存的是 seed100 的 **w=W_HI 单路**实测 wall；真正两路跑时每路 workers=5，
# 单 run 会更慢：两路 wall = 单路 wall × k。
# ★ v2 用逐环境 qflex 标定的 S 折算 k=2/S，实测**系统性偏乐观 8.6%**：
#   dacer 实测 k = 1.200 / 1.242 / 1.246（SMPL / h1-run / h1-balance），
#   轻算法 k 中位 1.28；即两路惩罚几乎与算法/环境无关 ≈1.24。
#   故 v3 改为统一常数，回代整批 29h07m vs 实际 29h04m（+0.17%）。
# ★ k 在同一环境内对所有算法相同 ⇒ 不改变 LPT 分配，仅修正预估数字。
k_of(){
  # 两路惩罚：实测统一 ≈1.24（dacer 1.23，轻算法 1.28~1.32）
  # 环境相关性被证伪 —— v2 的逐环境 S 折算反而更不准，见附录 E §11。
  printf '1.24'
}

# ── 构造 LPT 输入："alg:两路耗时 ..."（$1=env，$2=已过滤的算法列表） ──
spec_for(){
  local e="$1" K A C out=""
  K="$(k_of "$e")"
  for A in $2; do
    C="$(cost_of "$e" "$A")"
    out="$out $A:$(awk -v c="$C" -v k="$K" 'BEGIN{printf "%d", c*k}')"
  done
  printf '%s' "${out# }"
}

# ── 断点续跑探测：判据与 v5 完全一致（.formal_done + .formal_total 匹配） ──
is_done(){  # $1=env $2=alg $3=seed $4=suffix $5=tot
  local d
  for d in "$ROOT/logs/$1/$2"_*_s$3_$4; do
    [ -d "$d" ] || continue
    if [ -f "$d/.formal_done" ] && [ "$(cat "$d/.formal_total" 2>/dev/null)" = "$5" ]; then
      return 0
    fi
  done
  return 1
}

# ── LPT 分配：输入 "alg:cost alg:cost ..."，输出 "lane|alg alg ...|sumcost" ──
lpt(){
  printf '%s' "$1" | tr ' ' '\n' | sed '/^$/d' | awk -F: -v L="$2" '
    { a[NR]=$1; c[NR]=$2+0 }
    END{
      for(i=1;i<=NR;i++)for(j=i+1;j<=NR;j++)if(c[j]>c[i]){t=c[i];c[i]=c[j];c[j]=t;u=a[i];a[i]=a[j];a[j]=u}
      for(l=1;l<=L;l++)s[l]=0
      for(i=1;i<=NR;i++){b=1;for(l=2;l<=L;l++)if(s[l]<s[b])b=l;o[b]=o[b]" "a[i];s[b]+=c[i]}
      for(l=1;l<=L;l++){sub(/^ /,"",o[l]);printf "%d|%s|%d\n",l,o[l],s[l]}
    }'
}

# ── 计划展开（tmux 内外都要用） ──
# 每行：ENV|VENV|NENV|TOT|W_HI|SAVE
PLAN2=""
while IFS='|' read -r ENV VENV NENV TOT WHI; do
  [ -z "${ENV:-}" ] && continue
  case " $SKIP " in *" $ENV "*) continue;; esac
  if [ -n "$ONLY" ]; then
    case " $ONLY " in *" $ENV "*) : ;; *) continue;; esac
  fi
  if [ $((NENV % W_LO)) -ne 0 ]; then
    no "$ENV：W_LO=$W_LO 不整除 nenv=$NENV（process3.py:26 会 assert 失败）" >&2; exit 1
  fi
  SAVE=$(( TOT / 20 )); [ "$SAVE" -lt 1 ] && SAVE=1
  PLAN2="${PLAN2}${ENV}|${VENV}|${NENV}|${TOT}|${WHI}|${SAVE}
"
done <<< "$(printf '%s' "$TABLE2" | sed '/^$/d')"

N_ENVS="$(printf '%s' "$PLAN2" | grep -c . || true)"
[ "${N_ENVS:-0}" -eq 0 ] && { echo "[FAIL] 计划为空（检查 ONLY/SKIP）" >&2; exit 1; }
N_ALG=0; for a in $ALGS; do N_ALG=$((N_ALG+1)); done

# ── DRYRUN ──
if [ "$DRYRUN" = "1" ]; then
  say "DRYRUN：$N_ENVS 个环境 × $N_ALG 个算法 × seed(s) [$SEEDS]，每环境 $LANES 路并行（每路 workers=$W_LO）"
  echo "ALGS=[$ALGS]   WATCH_MB=$WATCH_MB   SWAP_MIN_MB=$SWAP_MIN_MB   GAP=${GAP}s"
  echo
  MISS=0
  while IFS='|' read -r ENV VENV NENV TOT WHI SAVE; do
    [ -z "${ENV:-}" ] && continue
    if [ -x "$ROOT/.venv/$VENV/bin/python" ]; then VP="venv OK"
    else VP="venv ** MISSING **"; MISS=$((MISS+1)); fi
    printf '%-26s %-22s %s\n' "$ENV" "$VENV" "$VP"
  done <<< "$(printf '%s' "$PLAN2" | sed '/^$/d')"
  [ "$MISS" -gt 0 ] && { no "有 $MISS 个 venv 缺失，先修 TABLE2" >&2; exit 1; }
  echo
  SER_ALL=0; PAR_ALL=0
  for S in $SEEDS; do
    SUF="s${S}"
    SER_S=0; PAR_S=0
    echo "──────── seed=$S  suffix=$SUF ────────"
    while IFS='|' read -r ENV VENV NENV TOT WHI SAVE; do
      [ -z "${ENV:-}" ] && continue
      TODO=""; SER=0
      for ALG in $ALGS; do
        if [ "$SKIP_DONE" = "1" ] && is_done "$ENV" "$ALG" "$S" "$SUF" "$TOT"; then
          continue
        fi
        TODO="$TODO $ALG"
        SER=$(( SER + $(cost_of "$ENV" "$ALG") ))
      done
      printf '%-26s nenv=%-3s total=%-9s W_HI=%-3s W_LO=%-3s → %s\n' \
        "$ENV" "$NENV" "$TOT" "$WHI" "$W_LO" "$( [ -z "$TODO" ] && echo '全部已完成，跳过（不占机时）' || echo "待跑:$TODO" )"
      [ -z "$TODO" ] && continue
      PAR=0
      while IFS='|' read -r L ASTR SC; do
        [ -z "${ASTR:-}" ] && continue
        [ "$SC" -gt "$PAR" ] && PAR="$SC"
        printf '      lane%s : %-42s 预估 %s（两路口径）\n' "$L" "$ASTR" "$(h2m "$SC")"
      done <<< "$(lpt "$(spec_for "$ENV" "$TODO")" "$LANES")"
      printf '      → 该格：串行 %s → 两路 %s（加速 %.2f×）\n' \
        "$(h2m "$SER")" "$(h2m "$PAR")" \
        "$(awk -v a="$SER" -v b="$PAR" 'BEGIN{ if(b>0) printf "%.2f", a/b; else printf "0" }')"
      SER_S=$(( SER_S + SER )); PAR_S=$(( PAR_S + PAR ))
    done <<< "$(printf '%s' "$PLAN2" | sed '/^$/d')"
    SER_ALL=$(( SER_ALL + SER_S )); PAR_ALL=$(( PAR_ALL + PAR_S ))
    echo "  ↳ seed=$S 小计：串行 $(h2m "$SER_S") → 两路 $(h2m "$PAR_S")"
    echo
  done
  if [ "$SER_ALL" -eq 0 ]; then
    echo "★★ 合计：本 seed 的 4 格**全部已完成**，没有可补的任务（不占机时）。"
    echo "    要看真实预估，请指定尚未完成的种子，例如： SEEDS=\"102\" DRYRUN=1 bash ~/run_envs_2lane_v3.sh"
  elif [ "$PAR_ALL" -lt "$SER_ALL" ]; then
    echo "★★ 合计（4 格，本脚本范围）：串行 $(h2m "$SER_ALL") → 两路 $(h2m "$PAR_ALL")，省 $(h2m $((SER_ALL-PAR_ALL)))"
  else
    echo "★★ 合计：两路 $(h2m "$PAR_ALL") **比串行 $(h2m "$SER_ALL") 更慢** —— LANES=$LANES 时降 workers 是纯亏（P≈1.09，见附录 E §8.1）"
  fi
  echo "   ⚠ 加速比 < S（1.58~1.84）是**必然**的：dacer 占单格 53~58% 时长、独占一条 lane，"
  echo "     另一条 lane 跑完 5 个轻算法就空转 ⇒ 上界是 max(lane) 而非 Σ/2。"
  echo "   ⚠ 不含 myoHand（两路实测爆内存）+ Ostrich（单路 ≈24.5 GB）—— 这两格用 v5 串行补。"
  echo "     ★ 千万别直接敲 'SEED=... bash ~/run_envs_formal_v5.sh'：v5 不向 tmux 传参，会静默回落"
  echo "       seed=100 ⇒ 全部已完成被跳过、脚本秒退 rc=0（看着像跑过了）。必须把变量写进"
  echo "       tmux 命令串，或加 NO_TMUX=1；正确写法见 交付附件 附录 E §9 / 记忆硬边界 18。"
  echo "       剩余：myoHand 只差 s102（s101 已完成）｜Ostrich 差 s101+s102（只能 w=2）。"
  exit 0
fi

# ── 前置检查 ──
[ -d "$ROOT" ] || { no "找不到 $ROOT" >&2; exit 1; }
if ! setsid --wait true >/dev/null 2>&1; then
  no "没有 setsid --wait —— 看门狗无法整组杀训练进程" >&2; exit 1
fi
while IFS='|' read -r ENV VENV NENV TOT WHI SAVE; do
  [ -z "${ENV:-}" ] && continue
  [ -x "$ROOT/.venv/$VENV/bin/python" ] || { no "缺 venv：$ROOT/.venv/$VENV/bin/python" >&2; exit 1; }
done <<< "$(printf '%s' "$PLAN2" | sed '/^$/d')"

if [ "$LANES" -lt 1 ]; then no "LANES 必须 ≥1"; exit 1; fi
if [ "$W_LO" -gt 20 ]; then warn "W_LO=$W_LO 偏大 —— 两路的内存是按 w=5 实测的，超了可能爆"; fi

# ── 独占守卫：已有训练在跑就拒绝（两路本身已到内存边缘） ──
if pgrep -f 'scripts/train\.py' >/dev/null 2>&1; then
  no "已有 scripts/train.py 在跑 —— 两路已吃满内存，拒绝叠加启动" >&2
  echo "     先等它收工，或：pkill -9 -f 'scripts/train.py'" >&2
  exit 1
fi

# ── 第 1 层：不在 tmux 内 → 自建会话后立刻返回 ──
if [ -z "${TMUX:-}" ] && [ "$NO_TMUX" != "1" ]; then
  command -v tmux >/dev/null 2>&1 || { no "没有 tmux，请先：sudo apt-get install -y tmux" >&2; exit 1; }
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    no "tmux 会话已存在：$SESSION（先 kill 或换 SESSION=）" >&2; exit 1
  fi
  SELF="$(readlink -f "$0")"
  # ★ 显式把变量写进 tmux 命令串（tmux 子进程不继承 prefix 变量 —— 这是 run_seeds_chain.sh 踩过的坑）
  tmux new-session -d -s "$SESSION" \
    "ALGS='$ALGS' SEEDS='$SEEDS' LANES='$LANES' W_LO='$W_LO' ONLY='$ONLY' SKIP='$SKIP' GAP='$GAP' \
WATCH_MB='$WATCH_MB' SWAP_MIN_MB='$SWAP_MIN_MB' RUN_TIMEOUT='$RUN_TIMEOUT' ROOT='$ROOT' LOGDIR='$LOGDIR' \
NO_TMUX=1 bash '$SELF'"
  ok "已在 tmux 会话 $SESSION 启动两路补种子"
  echo "     ALGS=[$ALGS]  SEEDS=[$SEEDS]  每环境 $LANES 路 × workers=$W_LO"
  echo "     覆盖：$(printf '%s' "$PLAN2" | cut -d'|' -f1 | tr '\n' ' ')"
  echo "     主日志：$MAINLOG"
  echo "     汇总  ：$SUM"
  echo "     启动前内存：total=$(mem_total)MB avail=$(mem_avail)MB swap=$(swap_total)MB"
  echo
  echo "  看进度 : tail -f $MAINLOG"
  echo "  attach : tmux attach -t $SESSION     （Ctrl-b 再按 d 退出，不影响训练）"
  echo "  中止   : tmux kill-session -t $SESSION"
  echo "           pkill -9 -f 'scripts/train.py'    # 训练走 setsid 独立进程组，须补这一刀"
  echo "  续跑   : 重跑同一条命令（已完成的 run 会自动跳过）"
  exit 0
fi

# ================= 以下在 tmux 会话内执行 =================
cd "$ROOT" || exit 1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true'
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PGDIR="$LOGDIR/.2lane_pg_$STAMP";   mkdir -p "$PGDIR"
SEGDIR="$LOGDIR/.2lane_seg_$STAMP"; mkdir -p "$SEGDIR"
LANEDIR="$LOGDIR/.2lane_spec_$STAMP"; mkdir -p "$LANEDIR"
PEAKFILE="$LOGDIR/.2lane_peak_$STAMP"; echo 0 > "$PEAKFILE"
KILLFILE="$LOGDIR/.2lane_kill_$STAMP"; : > "$KILLFILE"

{
  echo "===== 2-LANE BATCH START $(date '+%F %T') ====="
  echo "algs=[$ALGS] seeds=[$SEEDS] lanes=$LANES w_lo=$W_LO watchdog=${WATCH_MB}MB swap_min=${SWAP_MIN_MB}MB"
  echo "plan:"
  printf '%s' "$PLAN2" | sed '/^$/d' | sed 's/^/  /'
  echo "mem(before): total=$(mem_total)MB avail=$(mem_avail)MB swap=$(swap_total)MB"
  echo "★ 两路峰值已逐环境实测（附录 E §8）；但算法维度未标 ⇒ 看门狗 ${WATCH_MB}MB 是唯一护栏"
} | tee -a "$MAINLOG"

# ---------- 看门狗（真杀版，管所有 lane） ----------
(
  n=0
  while true; do
    sleep 2
    n=$((n+1))
    av="$(mem_avail)"; av=${av:-999999}
    sf="$(swap_free)"; sf=${sf:-999999}
    tot="$(mem_total)"; tot=${tot:-0}
    used=$((tot-av)); [ "$used" -lt 0 ] && used=0
    pk="$(cat "$PEAKFILE" 2>/dev/null)"; pk=${pk:-0}
    [ "$used" -gt "$pk" ] && echo "$used" > "$PEAKFILE"
    if [ $((n % 30)) -eq 0 ]; then
      printf '[mem %s] avail=%sMB swapfree=%sMB used=%sMB peak=%sMB\n' \
        "$(date '+%T')" "$av" "$sf" "$used" "$(cat "$PEAKFILE" 2>/dev/null)" >> "$MAINLOG"
    fi
    why=""
    if [ "$av" -lt "$WATCH_MB" ]; then
      why="MEM avail=${av}MB<${WATCH_MB}MB"
    elif [ "$sf" -lt "$SWAP_MIN_MB" ]; then
      why="SWAP swapfree=${sf}MB<${SWAP_MIN_MB}MB"
    else
      # 单 run 超时：逐个 lane 查
      for f in "$PGDIR"/*.start; do
        [ -f "$f" ] || continue
        st="$(cat "$f" 2>/dev/null)"; st=${st:-0}
        if [ "$st" -gt 0 ] && [ "$RUN_TIMEOUT" -gt 0 ] && \
           [ $(( $(date +%s) - st )) -gt "$RUN_TIMEOUT" ]; then
          why="TIMEOUT >${RUN_TIMEOUT}s"; break
        fi
      done
    fi
    if [ -n "$why" ]; then
      # ★ 只在 KILLFILE 为空时动手：否则每 2 s 会重复 kill 一遍、日志被刷屏
      if [ ! -s "$KILLFILE" ]; then
        echo "$why" > "$KILLFILE"
        echo "     [WATCHDOG] $why  ->  kill 所有 lane 的训练进程组" >> "$MAINLOG"
        for f in "$PGDIR"/*.pg; do
          [ -f "$f" ] || continue
          pg="$(cat "$f" 2>/dev/null)"
          [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
          : > "$f"
        done
      fi
    fi
  done
) &
WATCH_PID=$!

cleanup(){
  kill "$WATCH_PID" 2>/dev/null
  for f in "$PGDIR"/*.pg; do [ -f "$f" ] && : > "$f"; done
}
trap cleanup EXIT

echo "env|venv|alg|nenv|workers|total|save_every|rc|wall_s|peak_mb|kill_reason|rate|rundir" > "$SUM"

# ---------- 单条 lane 的执行体（在子进程里跑） ----------
lane_main(){   # $1 = lane 编号
  local LID="$1"
  local SPECF="$LANEDIR/lane${LID}.spec"
  [ -f "$SPECF" ] || return 0
  local line ENV VENV NENV TOT SAVE ALGSTR SUF
  line="$(cat "$SPECF")"
  IFS='|' read -r ENV VENV NENV TOT SAVE ALGSTR SUF <<< "$line"
  local VP="$ROOT/.venv/$VENV/bin/python"
  local PG="$PGDIR/lane${LID}.pg"
  local START="$PGDIR/lane${LID}.start"
  local SEG="$SEGDIR/lane${LID}_${ENV}.csv"
  : > "$SEG"
  for ALG in $ALGSTR; do
    local RLOG="$LOGDIR/2lane_${ENV}_${ALG}_s${SUF#s}_${STAMP}.log"
    {
      echo ""
      echo "--- [L$LID] $ENV / $ALG  开始 $(date '+%F %T')  avail=$(mem_avail)MB"
      echo "    日志：$RLOG"
    } >> "$MAINLOG"

    : > "$PG"; echo "$(date +%s)" > "$START"
    local T0; T0="$(date +%s)"
    setsid --wait bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PG" \
      "$VP" -u scripts/train.py \
      --alg "$ALG" --env "$ENV" --seed "${SUF#s}" \
      --total_step "$TOT" --num_vec_envs "$NENV" --suffix "$SUF" \
      --num_workers "$W_LO" --save_policy_every "$SAVE" \
      > "$RLOG" 2>&1 &
    local TP=$!
    wait "$TP"; local RC=$?
    : > "$PG"; echo 0 > "$START"
    local WALL=$(( $(date +%s) - T0 ))
    local PEAK; PEAK="$(cat "$PEAKFILE" 2>/dev/null)"; PEAK=${PEAK:-0}
    local KILLED; KILLED="$(cat "$KILLFILE" 2>/dev/null)"; KILLED="${KILLED//|/ }"
    [ -n "$KILLED" ] && RC=137
    local RD=""
    RD="$(ls -dt "$ROOT/logs/$ENV/${ALG}_"*_s"${SUF#s}"_"$SUF" 2>/dev/null | head -1)"
    local RATE
    RATE="$(awk -v t="$TOT" -v s="$WALL" 'BEGIN{ if (s>0) printf "%.0f", t/s; else printf "0" }')"
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
      "$ENV" "$VENV" "$ALG" "$NENV" "$W_LO" "$TOT" "$SAVE" "$RC" "$WALL" "$PEAK" "$KILLED" "$RATE" "$RD" >> "$SEG"

    if [ "$RC" -eq 0 ] && [ -n "$RD" ] && [ -d "$RD" ]; then
      printf '%s\n' "$TOT" > "$RD/.formal_total"
      touch "$RD/.formal_done"
      echo "[OK]   [L$LID] $ENV / $ALG 完成 $(h2m "$WALL")  peak=${PEAK}MB" >> "$MAINLOG"
    else
      echo "[FAIL] [L$LID] $ENV / $ALG 失败 rc=$RC $(h2m "$WALL") peak=${PEAK}MB ${KILLED:+（看门狗：$KILLED）}" >> "$MAINLOG"
    fi
    # 看门狗介入 = 该环境两路真的装不下 ⇒ 停本 lane（另一 lane 的失败由它自己记录）
    if [ -n "$KILLED" ]; then
      echo "[STOP] [L$LID] 看门狗已介入 ⇒ 停止本 lane 剩余任务（该环境两路不可行，改用 W_LO=2 复测或用 v5 串行）" >> "$MAINLOG"
      break
    fi
    sleep "$GAP"
  done
}

# ---------- 主循环：逐 seed × 逐环境 ----------
BATCH0=$(date +%s)
G_DONE=""; G_FAILED=""; G_SKIP_ENV=""
for S in $SEEDS; do
  SUF="s${S}"
  echo "" | tee -a "$MAINLOG"
  echo "########## SEED $S (suffix=$SUF) 开始 $(date '+%F %T') ##########" | tee -a "$MAINLOG"
  SEED0=$(date +%s)

  while IFS='|' read -r ENV VENV NENV TOT WHI SAVE; do
    [ -z "${ENV:-}" ] && continue
    : > "$KILLFILE"
    echo 0 > "$PEAKFILE"

    # 收集待跑算法 + LPT 预算
    TODO=""; N_SKIP=0
    for ALG in $ALGS; do
      if [ "$SKIP_DONE" = "1" ] && is_done "$ENV" "$ALG" "$S" "$SUF" "$TOT"; then
        N_SKIP=$((N_SKIP+1)); continue
      fi
      TODO="$TODO $ALG"
    done
    SPEC="$(spec_for "$ENV" "$TODO")"
    if [ -z "$TODO" ]; then
      say "[$ENV] seed=$S 全部已完成（$N_SKIP/$N_ALG）→ 跳过"
      echo "--- [$ENV] seed=$S 全部已完成 → 跳过" >> "$MAINLOG"
      G_SKIP_ENV="$G_SKIP_ENV ${ENV}/s${S}"
      continue
    fi
    say "[$ENV] seed=$S  nenv=$NENV total=$TOT  W_LO=$W_LO × $LANES 路  待跑:$TODO（已跳过 $N_SKIP）"
    echo "--- [$ENV] seed=$S 开始 $(date '+%F %T')  待跑:$TODO  avail=$(mem_avail)MB" | tee -a "$MAINLOG"

    # 写 lane spec
    # ★ lane 编号必须按**实际有任务**的 lane 收集 —— lpt 会给空 lane 也输出行，
    #   若用 `seq 1 N` 遍历，遇到「lane1 空、lane2 有任务」时任务会被整个丢掉。
    rm -f "$LANEDIR"/lane*.spec
    LANE_IDS=""
    while IFS='|' read -r L ASTR SC; do
      [ -z "${ASTR:-}" ] && continue
      LANE_IDS="$LANE_IDS $L"
      printf '%s|%s|%s|%s|%s|%s|%s\n' "$ENV" "$VENV" "$NENV" "$TOT" "$SAVE" "$ASTR" "$SUF" > "$LANEDIR/lane${L}.spec"
      echo "    lane$L : $ASTR   （预估 $(h2m "$SC")）" | tee -a "$MAINLOG"
    done <<< "$(lpt "$SPEC" "$LANES")"

    ENV0=$(date +%s)
    LPIDS=""
    for L in $LANE_IDS; do
      ( lane_main "$L" ) >> "$MAINLOG" 2>&1 &
      LPIDS="$LPIDS $!"
    done
    for p in $LPIDS; do wait "$p"; done
    ENVDT=$(( $(date +%s) - ENV0 ))

    # 汇总本环境
    N_OK=0; N_BAD=0
    for f in "$SEGDIR"/lane*_"$ENV".csv; do
      [ -f "$f" ] || continue
      while IFS='|' read -r e v a n w t sv rc wall peak kr rate rd; do
        [ -z "$e" ] && continue
        if [ "$rc" = "0" ]; then N_OK=$((N_OK+1)); G_DONE="$G_DONE ${e}/${a}";
        else N_BAD=$((N_BAD+1)); G_FAILED="$G_FAILED ${e}/${a}"; fi
      done < "$f"
    done
    KILLED_NOW="$(cat "$KILLFILE" 2>/dev/null)"
    echo "--- [$ENV] seed=$S 结束 $(date '+%F %T')  用时 $(h2m "$ENVDT")  成功 $N_OK  失败 $N_BAD  peak=$(cat "$PEAKFILE")MB ${KILLED_NOW:+【看门狗：$KILLED_NOW】}" | tee -a "$MAINLOG"
    if [ -n "$KILLED_NOW" ]; then
      warn "[$ENV] 两路内存不足（$KILLED_NOW）⇒ 该环境建议改用 W_LO=2 或 v5 串行"
    fi
    sleep "$GAP"
  done <<< "$(printf '%s' "$PLAN2" | sed '/^$/d')"

  echo "########## SEED $S 结束 $(date '+%F %T')  用时 $(h2m $(( $(date +%s) - SEED0 ))) ##########" | tee -a "$MAINLOG"
done

# ---------- 合并各 lane 的 CSV ----------
for f in "$SEGDIR"/lane*.csv; do
  [ -f "$f" ] && cat "$f" >> "$SUM"
done

# ---------- 结果自检 ----------
echo "" | tee -a "$MAINLOG"
say "结果自检（每格应有 $N_ALG 个算法：log.csv + .formal_done）"
{
  for S in $SEEDS; do
    SUF="s${S}"
    while IFS='|' read -r ENV VENV NENV TOT WHI SAVE; do
      [ -z "${ENV:-}" ] && continue
      C=0; M=""
      for ALG in $ALGS; do
        d="$(ls -dt "$ROOT/logs/$ENV/${ALG}_"*_s${S}_${SUF} 2>/dev/null | head -1)"
        if [ -n "$d" ] && [ -f "$d/.formal_done" ] && [ -f "$d/log.csv" ] && \
           [ "$(wc -l < "$d/log.csv")" -gt 1 ]; then
          C=$((C+1))
        else
          M="$M $ALG"
        fi
      done
      printf '  seed=%-4s %-26s %s/%s' "$S" "$ENV" "$C" "$N_ALG"
      if [ "$C" -lt "$N_ALG" ]; then printf '   [WARN] 未产出：%s\n' "${M:-（无）}"
      else printf '   OK\n'; fi
    done <<< "$(printf '%s' "$PLAN2" | sed '/^$/d')"
  done
} | tee -a "$MAINLOG"

{
  echo ""
  echo "===== 2-LANE BATCH DONE $(date '+%F %T') ====="
  echo "总线时：$(h2m $(( $(date +%s) - BATCH0 )))"
  echo "成功：${G_DONE:-（无）}"
  echo "失败：${G_FAILED:-（无）}"
  echo "整格跳过：${G_SKIP_ENV:-（无）}"
  echo "汇总：$SUM"
  echo ""
  echo "★ 还差两格（本脚本不含）——请用 v5 串行补："
  echo "  SEED=101 ONLY=\"myoHandPenTwirlRandom-v0\" bash ~/run_envs_formal_v5.sh"
  echo "  SEED=101 ONLY=\"OstrichRun-v1\" WITH_OSTRICH=1 WATCH_MB=1200 bash ~/run_envs_formal_v5.sh"
} | tee -a "$MAINLOG"

rm -rf "$LANEDIR" 2>/dev/null
