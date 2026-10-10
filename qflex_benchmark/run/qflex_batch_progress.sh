#!/usr/bin/env bash
# ============================================================================
# Qflex 批量 run 一览器
#
# 为什么需要它：批量跑基线时会有多个 run 目录，qflex_progress.sh 只显示最新
#   那一个。本脚本一次列出**所有** run 的真实步数 / ckpt / 评估末值 / 训练回报，
#   一屏看清哪个算法在跑、跑到哪、成绩如何。
#
# 用法：bash ~/qflex_batch_progress.sh
#       ENV_NAME=myoHandPose-v0 bash ~/qflex_batch_progress.sh   # 换环境
# ============================================================================
set -uo pipefail

ROOT="${ROOT:-$HOME/work/Qflex}"
ENV_NAME=${ENV_NAME:-myoLegWalk-v0}
VENV_PY="$ROOT/.venv/relax_myosuite/bin/python"
DIR="$ROOT/logs/$ENV_NAME"

if [ ! -d "$DIR" ]; then
  echo "[FAIL] 没有这个环境的日志目录：$DIR"
  exit 1
fi
if [ ! -x "$VENV_PY" ]; then
  echo "[FAIL] 找不到 venv python：$VENV_PY"
  exit 1
fi

echo "================================================================"
echo " 环境：$ENV_NAME"
echo " 目录：$DIR"
echo "================================================================"
echo "--- 正在跑的训练进程 ---"
FOUND=0
for P in /proc/[0-9]*; do
  CMD=$(tr '\0' ' ' < "$P/cmdline" 2>/dev/null) || continue
  case "$CMD" in
    *scripts/train.py*)
      A=$(echo "$CMD" | sed -n 's/.*--alg \([^ ]*\).*/\1/p')
      echo "  pid=$(basename "$P")  alg=${A:-?}"
      FOUND=1
      ;;
  esac
done
[ "$FOUND" -eq 0 ] && echo "  （没有在跑的训练进程 —— 可能全部跑完了）"

echo
echo "--- 各 run 状态（按最近写入排序）---"

cat > /tmp/_qflex_batch.py <<'PYEOF'
import sys, os, glob, struct, csv, re, time
from tensorboardX.proto.event_pb2 import Event

TAGS = ("sample/episode_return", "sample/episode_length")

def parse_events(path):
    last_step = -1
    vals = {}
    try:
        with open(path, "rb") as f:
            while True:
                hdr = f.read(12)
                if len(hdr) < 12:
                    break
                (ln,) = struct.unpack("<Q", hdr[:8])
                data = f.read(ln)
                if len(data) < ln:
                    break
                f.read(4)
                ev = Event()
                try:
                    ev.ParseFromString(data)
                except Exception:
                    continue
                if not ev.HasField("summary"):
                    continue
                if ev.step > last_step:
                    last_step = ev.step
                for v in ev.summary.value:
                    if v.tag in TAGS and v.HasField("simple_value"):
                        vals[v.tag] = v.simple_value
    except Exception as e:
        vals["__err__"] = "%s: %s" % (type(e).__name__, e)
    return last_step, vals

def read_total(run):
    p = os.path.join(run, "config.yaml")
    try:
        for l in open(p):
            m = re.match(r"total_step:\s*(\d+)", l)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 0

def eval_last(run):
    p = os.path.join(run, "log.csv")
    if not os.path.exists(p):
        return None, 0
    try:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None, 0
    if not rows:
        return None, 0
    last = rows[-1]
    for k in ("avg_ret", "avg_return", "mean_ret", "avg_reward", "return"):
        if k in last and last[k] not in ("", None):
            try:
                return float(last[k]), len(rows)
            except ValueError:
                pass
    return None, len(rows)

print("%-6s %-40s %13s %5s %5s %12s %11s" %
      ("ALG", "RUN", "STEP", "TOT%", "CKPT", "EVAL_LAST", "TRAIN_RET"))
print("-" * 98)

for run in sys.argv[1:]:
    run = run.rstrip("/")
    name = os.path.basename(run)
    alg = name.split("_")[0]
    tfs = glob.glob(os.path.join(run, "events.out.tfevents.*"))
    step, vals = (-1, {})
    if tfs:
        tfs.sort(key=os.path.getmtime)
        step, vals = parse_events(tfs[-1])
    total = read_total(run)
    ck = len(glob.glob(os.path.join(run, "policy-*.pkl")))
    ev, nev = eval_last(run)

    pct = ("%.0f%%" % (100.0 * step / total)) if (total > 0 and step >= 0) else "-"
    step_s = ("%d" % step) if step >= 0 else "-"
    tot_s = ("%d" % total) if total > 0 else "-"
    ev_s = ("%.1f" % ev) if ev is not None else ("n=%d" % nev if nev else "-")
    tr = vals.get("sample/episode_return")
    tr_s = ("%.1f" % tr) if tr is not None else "-"

    # 活跃度：最新 events 的写入时间
    age = ""
    if tfs:
        age = "%d min 前" % int((time.time() - os.path.getmtime(tfs[-1])) / 60)
        if age.startswith("0 min"):
            age = "刚刚"
    print("%-6s %-40s %13s %5s %5s %12s %11s" %
          (alg[:6], name[:40], ("%s/%s" % (step_s, tot_s)), pct, ck, ev_s, tr_s))
    if age:
        print("       └ 最后写入：%s" % age)

if len(sys.argv) > 1:
    print()
    print("提示：EVAL_LAST = log.csv 的评估值（论文口径，每 total/20 步一次）；")
    print("      TRAIN_RET = 训练期采样回报（看训练健康度）。两者语义不同，别混。")
PYEOF

# shellcheck disable=SC2046
"$VENV_PY" /tmp/_qflex_batch.py $(ls -dt "$DIR"/*/ 2>/dev/null) 2>&1 | head -60

echo
echo "--- 资源 ---"
awk '/MemTotal/{t=int($2/1024)} /MemAvailable/{a=int($2/1024)} /SwapTotal/{st=int($2/1024)} /SwapFree/{sf=int($2/1024)} END{printf "  内存 : 可用 %d / %d MB   swap 剩余 %d / %d MB\n", a, t, sf, st}' /proc/meminfo
echo "  磁盘 : $(df -h "$HOME" | awk 'NR==2{printf "已用 %s / 共 %s (%s)", $3, $2, $5}')"
NSMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
if [ -x "$NSMI" ]; then
  "$NSMI" --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
          --format=csv,noheader 2>/dev/null | sed 's/^/  GPU  : /'
fi
