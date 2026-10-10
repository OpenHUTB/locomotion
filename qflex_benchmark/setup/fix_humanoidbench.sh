#!/usr/bin/env bash
# =====================================================================
# fix_humanoidbench.sh
#   目标：把 .venv/relax_humanoidbench 的依赖装完
#         （对应环境 h1-run-v0 / h1-balance_simple-v0）
#
#   ── 冲突诊断（可逐条复核，全是硬证据）──
#   1) requirements/uv/relax_humanoidbench.txt 内容 = `-r base.txt` + `-e envs/humanoid-bench`
#   2) envs/humanoid-bench/setup.py:18   install_requires 里硬钉 torch==2.3.1
#      PyPI 元数据：torch 2.3.1 在 linux/x86_64 上要求 nvidia-cudnn-cu12==8.9.2.26
#   3) 我们必须用 jax 0.4.38（0.4.27 在本机 sm_120 上 GPU 起不来）
#      PyPI 元数据：jax-cuda12-plugin 0.4.38 的 with-cuda extra 要求 nvidia-cudnn-cu12>=9.1,<10
#   4) 8.9.2.26 与 >=9.1 互斥 → uv 判定 unsatisfiable → 整个 venv 的依赖一个都装不上
#   ★ 注意：jax-cuda12-plugin 0.4.27 的元数据里 **没有** with-cuda 这段 nvidia 约束
#     ⇒ 这个冲突是"我们为了 Blackwell 把 jax 从 0.4.27 升到 0.4.38"才引入的，
#       不是原作者的环境坏了。（所以别去改 requirements 文件，改的是安装参数。）
#
#   ── 修法 ──
#   给这个 venv 单独加 `--torch-backend=cpu`
#     → torch 解析成 2.3.1+cpu（PyTorch CPU 索引的 wheel，不依赖任何 nvidia-* 包）
#     → cudnn 冲突消失，jax 仍保持 CUDA 版（不动）
#   为什么安全：envs/humanoid-bench/humanoid_bench/env.py 全文只 import
#     os/numpy/mujoco/gymnasium/dm_control + 自己的 dmc_deps，**不 import torch**；
#     torch 只被 humanoid-bench 自带的 baseline 算法用到，而 relax 训练走 jax。
#     H1 环境是 MuJoCo CPU 仿真，CPU 版 torch 完全够用。
#
#   用法：      bash ~/fix_humanoidbench.sh
#   只看解析：  DRY_ONLY=1 bash ~/fix_humanoidbench.sh
# =====================================================================
set -uo pipefail

QROOT="${QROOT:-$HOME/work/Qflex}"
NAME="relax_humanoidbench"
V="$QROOT/.venv/$NAME"
REQ_REL="requirements/uv/relax_humanoidbench.txt"
JAX_RECORD="$QROOT/.jax_version_ok"
PROBE_PY="/tmp/_jaxprobe2.py"
DRY_LOG="/tmp/_hb_dry.log"
GCC_RELAX="-O2 -fPIC -DNDEBUG -std=gnu11 -Wno-error -Wno-incompatible-pointer-types"   # 与 setup_other_envs_v2.sh:95 同源
DRY_ONLY="${DRY_ONLY:-0}"

C_GO=$'\033[1;32m'; C_NO=$'\033[1;31m'; C_WA=$'\033[1;33m'; C_TI=$'\033[1;36m'; C_0=$'\033[0m'
say(){  printf '\n%s===== %s =====%s\n' "$C_TI" "$1" "$C_0"; }
ok(){   printf '%s[OK]%s   %s\n'   "$C_GO" "$C_0" "$1"; }
no(){   printf '%s[FAIL]%s %s\n'   "$C_NO" "$C_0" "$1"; }
warn(){ printf '%s[WARN]%s %s\n'   "$C_WA" "$C_0" "$1"; }
info(){ printf '       %s\n' "$1"; }

# GPU 探针（与 setup_other_envs_v2.sh / setup_qflex.sh 同源，判据一致：
#   唯一通过标准 = 末尾出现 PROBE_RESULT: OK）
write_probe() {
  cat > "$PROBE_PY" <<'PYEOF'
import os, math, traceback
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
ok = False
try:
    import jax, jax.numpy as jnp, numpy as np
    print("jax_version:", jax.__version__)
    d = jax.devices()
    gpu = [x for x in d if x.platform == "gpu"]
    print("devices:", [str(x) for x in d], "| GPU_COUNT:", len(gpu))
    print("NAMED_SHAPE_OK:", hasattr(jax.ShapeDtypeStruct((1,), "float32"), "named_shape"))
    if gpu:
        print("DEVICE_KIND:", gpu[0].device_kind)
        a = jnp.ones((1024, 1024), dtype=jnp.float32)
        s = (a @ a).sum(); s.block_until_ready(); v = float(s)
        if math.isfinite(v):
            print("MATMUL_OK:", v); ok = True
        else:
            print("MATMUL_BAD:", v)
except Exception as e:
    traceback.print_exc(); print("PROBE_ERR:", type(e).__name__, str(e)[:300])
print("PROBE_RESULT:", "OK" if ok else "FAIL")
PYEOF
}
run_probe() { [ -f "$PROBE_PY" ] || write_probe; "$1/bin/python" "$PROBE_PY" 2>&1; }
probe_ver() { printf '%s\n' "$1" | sed -n 's/^jax_version: *//p' | tail -1; }
probe_ok()  { printf '%s\n' "$1" | grep -qE "^PROBE_RESULT: *OK"; }

# ---------------------------------------------------------------------
say "0) 前置检查"
export PATH="$HOME/.local/bin:$PATH"
cd "$QROOT" 2>/dev/null || { no "找不到项目目录 $QROOT（可用 QROOT=... 指定）"; exit 1; }
info "项目目录 : $(pwd)"
command -v uv >/dev/null 2>&1 && ok "uv : $(uv --version)" || { no "没有 uv —— 先跑 bash ~/setup_qflex.sh"; exit 1; }
[ -x "$V/bin/python" ] || { no "没有 $V/bin/python —— 先跑 bash ~/setup_other_envs_v2.sh"; exit 1; }
ok "venv     : $("$V/bin/python" -V 2>&1)"
[ -f "$REQ_REL" ] || { no "找不到 $REQ_REL"; exit 1; }

SPEC=""
[ -s "$JAX_RECORD" ] && SPEC="$(head -n1 "$JAX_RECORD" | tr -d '[:space:]')"
SPEC="${SPEC:-jax[cuda12]==0.4.38}"
info "jax spec : $SPEC   （沿用已实测通过的版本，不重挑）"

# ---------------------------------------------------------------------
say "1) 干跑解析（--dry-run + --torch-backend=cpu，先不安装）"
info "命令: uv pip install --python .venv/$NAME --dry-run --torch-backend=cpu $SPEC -r $REQ_REL"
info "带 CFLAGS 兜底：dry-run 也要为 -e . 生成 editable 元数据，会编译 src/futex.c"
if CFLAGS="$GCC_RELAX" uv pip install --python "$V" --dry-run --torch-backend=cpu "$SPEC" -r "$REQ_REL" > "$DRY_LOG" 2>&1; then
  ok "解析通过 —— 冲突已解"
  echo "  ---- 解析结果（torch / jax / gymnasium / mujoco 等关键包）----"
  grep -oE "\b(torch|torchvision|jax|jaxlib|jax-cuda12-plugin|jax-cuda12-pjrt|nvidia-cudnn-cu12|numpy|gymnasium|mujoco|mujoco-mjx|brax|gymnax|flax|dm-control|dm_control)[=<>!~]+[0-9][^ ,;)]*" "$DRY_LOG" \
    | sort -u | sed 's/^/    /'
  echo "  ---- 末尾 ----"
  tail -6 "$DRY_LOG" | sed 's/^/    /'
else
  no "干跑就失败了 → 完整日志：$DRY_LOG"
  tail -40 "$DRY_LOG" | sed 's/^/    /'
  warn "把上面这段发我，别继续装。"
  exit 1
fi

if [ "$DRY_ONLY" = "1" ]; then
  say "DRY_ONLY=1 → 到此为止，什么都没装"
  info "真装请跑: bash ~/fix_humanoidbench.sh"
  exit 0
fi

# ---------------------------------------------------------------------
say "2) 正式安装（torch 走 CPU 索引；src/futex.c 用放宽的 CFLAGS 兜底）"
info "说明：第一次失败多半是 GCC 对 src/futex.c 的 incompatible-pointer-types 报错，"
info "      脚本会自动放宽诊断等级重试，红色报错可忽略。"
if uv pip install --python "$V" --torch-backend=cpu "$SPEC" -r "$REQ_REL"; then
  ok "依赖安装完成（一次过）"
elif CFLAGS="$GCC_RELAX" uv pip install --python "$V" --torch-backend=cpu "$SPEC" -r "$REQ_REL"; then
  ok "依赖安装完成（CFLAGS 放宽后）"
elif CFLAGS="-O2 -fPIC -w" uv pip install --python "$V" --torch-backend=cpu "$SPEC" -r "$REQ_REL"; then
  ok "依赖安装完成（-w 抑制全部警告后）"
else
  no "依赖安装失败 —— 把上面的完整输出发我"
  exit 1
fi

# ---------------------------------------------------------------------
say "3) 复验：jax 有没有被依赖顶掉 + GPU 还在不在"
OUT="$(run_probe "$V")"
printf '%s\n' "$OUT" | sed 's/^/    /'
VER="$(probe_ver "$OUT")"
if probe_ok "$OUT" && [ "jax[cuda12]==$VER" = "$SPEC" -o "jax==$VER" = "$SPEC" ]; then
  ok "jax 仍是 $VER 且 GPU 可用（没被 torch/humanoid-bench 顶掉）"
else
  warn "复验不符（选定 $SPEC / 实到 ${VER:-读不到}）→ 把 jax 装回去"
  uv pip install --python "$V" "$SPEC" >/dev/null 2>&1
  OUT="$(run_probe "$V")"
  if probe_ok "$OUT"; then ok "回滚后 GPU 通过（jax $(probe_ver "$OUT")）"
  else no "回滚也没修好 —— 把上面的输出发我"; exit 1; fi
fi

# ---------------------------------------------------------------------
say "4) 环境自检（h1-run-v0 / h1-balance_simple-v0）"
info "预期: h1-run-v0 |S|=51 |A|=19（论文 Figure 2(a)）"
info "      h1-balance_simple-v0 **论文维度表里没有**，只能用同机器人 H1 的 51/19 猜（推断，脚本不比对）"
"$V/bin/python" - <<'PYEOF'
import warnings; warnings.filterwarnings("ignore")
try:
    import envs
except Exception as e:
    print(f"  (warn) import envs: {type(e).__name__}: {e}")
try:
    import humanoid_bench      # 注册 h1-<task>-v0
except Exception as e:
    print(f"  (warn) import humanoid_bench: {type(e).__name__}: {e}")
import gymnasium as gym, traceback
for name in ("h1-run-v0", "h1-balance_simple-v0"):
    try:
        try:
            e = gym.make(name, render_mode="rgb_array")   # relax 的第一选择
        except Exception:
            e = gym.make(name)                             # relax 的回退
        print(f"  [OK]   {name:24s} |S|={e.observation_space.shape[0]:5d} |A|={e.action_space.shape[0]:4d}")
        e.close()
    except Exception:
        print(f"  [FAIL] {name}"); traceback.print_exc()
PYEOF

say "完成"
info "下一步（第 2 步冒烟）记住必须显式给 --save_policy_every，否则 3e5 步会触发 20 次评估"
info "  cd ~/work/Qflex"
info "  export XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false"
info "  .venv/relax_humanoidbench/bin/python scripts/train.py --alg qflex --env h1-run-v0 --seed 100 --total_step 300000 --num_vec_envs 70 --save_policy_every 300000 --suffix smoke"
