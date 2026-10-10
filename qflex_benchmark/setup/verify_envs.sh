#!/usr/bin/env bash
# =====================================================================
# verify_envs.sh
#   一条命令验证 4 套 venv / 7 个环境能否 import + 建环境，并打印 |S| |A|
#   （对论文 Figure 2(a) 的维度表自检）
#
#   用法:  bash ~/verify_envs.sh
#
#   ★ 修正 setup_other_envs_v2.sh 里的判据 bug ★
#   v2 的 check_env 只 `import envs`（QFlex 自注册的 -v1 系列），
#   没有 import 提供 myo*-v0 / h1-*-v0 的两个包 → 报 NameNotFound。
#   证据：
#     .venv/relax_myosuite/.../myosuite/utils/__init__.py:22  gym = import_gym()  # 返回 gymnasium
#     myosuite/__init__.py                                    把 myoLegWalk-v0 等注册进 gymnasium 注册表
#     relax/env/__init__.py:8                                 import myosuite  ← 训练时靠这一行注册
#     envs/humanoid-bench/humanoid_bench/__init__.py          把 h1-<task>-v0 注册进 gymnasium
#   ⇒ 那次 h1-run-v0 / h1-balance_simple-v0 的 FAIL 是真失败（依赖没装上），
#     myoLegWalk-v0 / myoHandPenTwirlRandom-v0 的 FAIL 是假失败（脚本没 import）。
#
#   判据与训练同源：优先用 relax.env.create_env()（训练真正走的入口），
#   拿不到再退回 gym.make + space.shape。
# =====================================================================
set -uo pipefail

QROOT="${QROOT:-$HOME/work/Qflex}"

C_GO=$'\033[1;32m'; C_NO=$'\033[1;31m'; C_WA=$'\033[1;33m'; C_TI=$'\033[1;36m'; C_0=$'\033[0m'
say(){  printf '\n%s===== %s =====%s\n' "$C_TI" "$1" "$C_0"; }
ok(){   printf '%s[OK]%s   %s\n'   "$C_GO" "$C_0" "$1"; }
no(){   printf '%s[FAIL]%s %s\n'   "$C_NO" "$C_0" "$1"; }
warn(){ printf '%s[WARN]%s %s\n'   "$C_WA" "$C_0" "$1"; }
info(){ printf '       %s\n' "$1"; }

# 论文 Figure 2(a) 的维度表（本工作列）
EXPECT_S="myoLegWalk-v0:403 myoHandPenTwirlRandom-v0:83 SMPLHumanoidJump-v1:358 OstrichRun-v1:596 MS700Locomotion-v1:3148 h1-run-v0:51"
EXPECT_A="myoLegWalk-v0:80 myoHandPenTwirlRandom-v0:39 SMPLHumanoidJump-v1:69 OstrichRun-v1:120 MS700Locomotion-v1:700 h1-run-v0:19"

cd "$QROOT" 2>/dev/null || { no "找不到项目目录 $QROOT（可 QROOT=... 指定）"; exit 1; }
say "验证 7 个环境（项目目录 $(pwd)）"
info "预期 |S| : $EXPECT_S"
info "预期 |A| : $EXPECT_A"

FAILED=0
PASSED=0

probe_one() {   # $1=venv 名  $2=环境 id
  local V="$QROOT/.venv/$1" NAME="$2" OUT="" rc=0
  if [ ! -x "$V/bin/python" ]; then warn "跳过 $NAME（.venv/$1 不存在）"; FAILED=$((FAILED+1)); return; fi
  OUT="$("$V/bin/python" - "$NAME" "$EXPECT_S" "$EXPECT_A" <<'PYEOF'
import sys, warnings, traceback
warnings.filterwarnings("ignore")
name, exp_s, exp_a = sys.argv[1], sys.argv[2], sys.argv[3]

def look(tbl, k):
    for tok in tbl.split():
        i, _, v = tok.partition(":")
        if i == k:
            return int(v)
    return None

# 1) 三个注册来源，缺哪个都会 NameNotFound
try:
    import envs                      # QFlex 自注册：OstrichRun-v1 / MS700Locomotion-v1 / SMPLHumanoidJump-v1
except Exception as e:
    print(f"  (warn) import envs 失败: {type(e).__name__}: {e}")
for m in ("myosuite", "humanoid_bench"):   # myo*-v0 / h1-*-v0
    try:
        __import__(m)
    except Exception as e:
        print(f"  (warn) import {m} 失败: {type(e).__name__}: {e}")

import gymnasium as gym
try:
    # 与训练入口同源：relax/env/__init__.py:92-104 create_env()
    from relax.env import create_env
    env, s, a = create_env(name, seed=100)
    src = "create_env"
except Exception:
    try:
        try:
            env = gym.make(name, render_mode="rgb_array")   # relax 的第一选择
        except Exception:
            env = gym.make(name)                             # relax 的回退
        s = env.observation_space.shape[0]
        a = env.action_space.shape[0]
        src = "gym.make"
    except Exception:
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        sys.exit(3)

es, ea = look(exp_s, name), look(exp_a, name)
bad = (es is not None) and not (s == es and a == ea)
tag = f"MISMATCH (论文 {es}/{ea})" if bad else "MATCH"
print(f"  {'[FAIL]' if bad else '[OK]  '} {name:26s} |S|={s:5d} |A|={a:4d}   {tag}   [{src}]")
try:
    env.close()
except Exception:
    pass
sys.exit(3 if bad else 0)
PYEOF
)"
  rc=$?
  printf '%s\n' "$OUT"
  if [ $rc -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
}

probe_one relax_myosuite      myoLegWalk-v0
probe_one relax_myosuite      myoHandPenTwirlRandom-v0
probe_one relax_humanoidbench h1-run-v0
probe_one relax_humanoidbench h1-balance_simple-v0
probe_one relax_ms            OstrichRun-v1
probe_one relax_ms            MS700Locomotion-v1
probe_one relax_smplhumanoid  SMPLHumanoidJump-v1

say "汇总"
echo "  通过 / 失败   : $PASSED / $FAILED"
echo "  venv 列表     : $(ls "$QROOT/.venv" 2>/dev/null | tr '\n' ' ')"
echo "  submodule     :"
for d in envs/humanoid-bench envs/ostrich/ostrichrl envs/ms_700_walk/MS-Human-700; do
  printf '    %-38s %s\n' "$d" "$(git -C "$QROOT/$d" rev-parse --short=12 HEAD 2>/dev/null || echo '(不是 git 仓库)')"
done
info "(warn) 行是正常噪声：某些 venv 里没有别的环境组的包，属预期"
