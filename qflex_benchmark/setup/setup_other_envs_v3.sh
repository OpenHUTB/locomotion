# =====================================================================
# ★ v3 = v2 + 两处修正（v2 的两个 bug 已在 setup_other_envs_v3.sh 里修好）
#   修正 A（build_venv / 2.3 安装段）：relax_humanoidbench 加 --torch-backend=cpu
#     —— 否则 torch 2.3.1 要 nvidia-cudnn-cu12==8.9.2.26 与 jax 0.4.38 要 >=9.1 互斥，
#        uv 报 unsatisfiable，该 venv 依赖一个都装不上（v2 实测踩到）。
#   修正 B（check_env 探针）：补 `import myosuite` / `import humanoid_bench`
#     —— 否则 myoLegWalk-v0 / myoHandPenTwirlRandom-v0 假 FAIL（v2 实测踩到）。
#   另有：--torch-backend 只在 h1 组生效，其余 venv 行为与 v2 完全一致；
#         已建好的 venv 会走"venv 已存在"分支，不会重装。
#   若只想补 h1 那一套，用同目录的 fix_humanoidbench.sh（更快，不重跑全流程）。
# =====================================================================
#!/usr/bin/env bash
# =====================================================================
# Qflex 其余环境组搭建 —— WSL / Ubuntu 一键脚本   v2   (2026-09-26)
#
# 目标：为论文 Figure 2(b) 剩下的环境格准备环境。共 3 套 venv + 3 个 submodule。
#
# 环境 → venv → 依赖 对照表（逐行核过官方文件，不是照 README 的例子猜的）：
#   venv                    环境                              requirements
#   .venv/relax_humanoidbench  h1-run-v0 / h1-balance_simple-v0  relax_humanoidbench.txt
#   .venv/relax_smplhumanoid   SMPLHumanoidJump-v1               relax_smplhumanoid.txt
#   .venv/relax_ms             OstrichRun-v1 / MS700Locomotion-v1 relax_ms_700_walk.txt
#   .venv/relax_myosuite       myoLegWalk-v0 / myoHandPenTwirlRandom-v0  【已建，本脚本不动】
#
# ★ 为什么不能照 README 抄：
#   README.md:126-132 举的例子写 "-e envs/ostrich/ostrichrl"，但实际 requirements
#   文件里**没有这一行**（全仓库 grep '^-e' 只有两处：base.txt 的 "-e ."、
#   relax_humanoidbench.txt 的 "-e envs/humanoid-bench"）。
#   ostrichrl / MS-Human-700 是**当资源目录用**的，不需要 pip 安装：
#     envs/ostrich/OstrichRun_v1.py:41     读 .../ostrichrl/ostrichrl/assets/models/ostrich/ostrich.xml
#     envs/ms_700_walk/locomotion_v1.py:43 读 .../MS-Human-700/MS-Human-700.xml
#   结论：只有 humanoid-bench 必须 clone **且** pip install -e。
#
# ★ 为什么必须 clone 指定 commit 而不是默认分支：
#   我们是 zip 下载的（不是 git clone），zip **不包含 submodule 内容，也不包含
#   作者锁定的 commit 号**（pin 存在 git tree 的 gitlink 里，.gitmodules 只有 url）。
#   下面三个 SHA 是 2026-09-26 从 GitHub API 读到的**作者在 main 上锁定的版本**，
#   直接检出它们才是"复现作者那次实验"，检默认分支可能已经漂移。
#
# ★★ 最大的坑：jax 版本
#   requirements/uv/jax-cuda12.txt:3  锁的是 jax[cuda12]==0.4.27
#   —— 这个版本在本机（RTX 5070 Ti Laptop / sm_120 / Blackwell）上
#      **任何 GPU 算子都编译不了**（jaxlib 0.4.27 自带 ptxas 只认到 sm_90a）：
#        ptxas fatal : Program with .target 'sm_90a' cannot be compiled to
#        future architecture
#   而且 base.txt 里 optax / numpyro **都没锁版本** —— 不把 jax spec 一起交给
#   resolver 的话，装依赖会立刻把 jax 顶到 0.7+（optax 最新要 jax>=0.5.3，
#   numpyro 最新要 jax>=0.7.0）。
#   ⇒ 本脚本对每套新 venv：优先沿用 relax_myosuite 已实测通过的 spec（读
#     $QROOT/.jax_version_ok），否则用 0.4.38，且**每次安装都把 jax spec 一起传给
#     resolver**，装完再复验一次。
#
# ★ 不用重打 persistence.py 补丁：
#   该补丁改的是 $QROOT/relax/utils/persistence.py（项目源码），而所有 venv 都用
#   "-e ." 可编辑安装指向**同一份源码** ⇒ 补丁天然共享。本脚本只做存在性检查，
#   不重复打（避免对已打过补丁的文件再动一次）。C 扩展（src/futex.c 等）则
#   每套 venv 各自编译 —— 故保留了 Ubuntu 26.04 / GCC 14+ 的诊断等级回退。
#
# 用法（在 qflex_benchmark/ 目录下执行；路径可用 QROOT 覆盖）:
#   QROOT=~/work/Qflex bash setup/setup_other_envs_v3.sh 2>&1 | tee ~/setup_other_envs.log
#   （仓库内文件已是 LF 行尾，直接跑即可；仅当你从 Windows 手动拷到 WSL 时才需要：
#      tr -d '\r' < setup_other_envs_v3.sh > ~/setup_other_envs_v3.sh）
#
#   分段执行:   bash ~/setup_other_envs.sh submodules | envs | verify
#   只建一套:   ONLY=relax_ms bash ~/setup_other_envs.sh envs
#   强制重探 jax: FORCE_JAX_PROBE=1 bash ~/setup_other_envs.sh envs
#
# 幂等：可重复运行，已完成的会自动跳过。
# =====================================================================

MODE="${1:-all}"
QROOT="${QROOT:-$HOME/work/Qflex}"
PYVER="${PYVER:-3.12}"
ONLY="${ONLY:-}"

JAX_RECORD="$QROOT/.jax_version_ok"       # 已实测通过的 jax spec（relax_myosuite 写的）
JAX_REJECTED="$QROOT/.jax_rejected"       # 本机探针实测失败的 spec
PROBE_PY="/tmp/_jaxprobe2.py"
export UV_CACHE_DIR="$QROOT/.uv-cache"    # ★ 三套 venv 共用缓存 → wheel 只下一次
export UV_NO_PROGRESS=1

# ★ v2 修复：归档 .jax_rejected
#   v1 因探针文件从未生成，把**所有** jax 候选（含实测可用的 0.4.38）都误判为 FAIL
#   并写入这里 ⇒ 不清理的话 v2 会继续跳过正确版本。改名而非删除，保留取证。
if [ -f "$JAX_REJECTED" ]; then
  mv "$JAX_REJECTED" "$JAX_REJECTED.bogus.$(date +%Y%m%d-%H%M%S)" 2>/dev/null \
    && printf '[v3] 已归档误判黑名单 %s\n' "$JAX_REJECTED"
fi

CH=$'\033[1;36m'; CO=$'\033[1;32m'; CN=$'\033[1;31m'; CY=$'\033[1;33m'; CE=$'\033[0m'
say() { printf '\n%s===== %s =====%s\n' "$CH" "$*" "$CE"; }
ok()  { printf '%s[OK]%s   %s\n' "$CO" "$CE" "$*"; }
no()  { printf '%s[FAIL]%s %s\n' "$CN" "$CE" "$*"; }
warn(){ printf '%s[WARN]%s %s\n' "$CY" "$CE" "$*"; }
info(){ printf '       %s\n' "$*"; }

# ---- 作者锁定的 submodule commit（2026-09-26 取自 GitHub API 的 main 树）----
SUBM_HB_SHA="cb1189039151c8aadaaa987b442da54383c87fab"
SUBM_OS_SHA="206a6038bf0e67850b7f3fb3f16d74b7ff88f010"
SUBM_MS_SHA="2d686957aefd5739cf4d2859a6acfd94c8f84400"
SUBM_HB_URL="https://github.com/carlosferrazza/humanoid-bench.git"
SUBM_OS_URL="https://github.com/vittorione94/ostrichrl.git"
SUBM_MS_URL="https://github.com/LNSGroup/MS-Human-700.git"

# GCC 14+ 把 src/futex.c 的指针类型不匹配从 warning 提成 error；这是诊断等级问题，
# 不是 C 标准问题（CFLAGS='-std=gnu11' 实测无效）。不改作者源码，只放诊断等级。
GCC_RELAX="-O2 -fPIC -DNDEBUG -std=gnu11 -Wno-error -Wno-incompatible-pointer-types"

# 论文 Figure 2(a) 的维度（用于装完自检；本机实测 myoLeg=403/80、myoHand=83/39 已核对一致）
EXPECT_S="myoLegWalk-v0:403 myoHandPenTwirlRandom-v0:83 SMPLHumanoidJump-v1:358 OstrichRun-v1:596 MS700Locomotion-v1:3148 h1-run-v0:51"
EXPECT_A="myoLegWalk-v0:80 myoHandPenTwirlRandom-v0:39 SMPLHumanoidJump-v1:69 OstrichRun-v1:120 MS700Locomotion-v1:700 h1-run-v0:19"

# ---------------------------------------------------------------------
say "0) 前置检查"
export PATH="$HOME/.local/bin:$PATH"
cd "$QROOT" 2>/dev/null || { no "找不到项目目录 $QROOT（用 QROOT=... 指定）"; exit 1; }
info "项目目录     : $(pwd)"
command -v uv  >/dev/null 2>&1 && ok "uv  : $(uv --version)"  || { no "没有 uv —— 先跑 bash ~/setup_qflex.sh"; exit 1; }
command -v git >/dev/null 2>&1 && ok "git : $(git --version)" || { no "没有 git"; exit 1; }
info "uv 缓存      : $UV_CACHE_DIR（三套 venv 共用，wheel 只下一次）"
if [ -f "$JAX_RECORD" ]; then
  info "已记录 jax   : $(head -n1 "$JAX_RECORD")   ← 新 venv 会优先沿用它"
else
  warn "没有 $JAX_RECORD → 新 venv 将从 jax[cuda12]==0.4.38 开始试"
fi

# ---------------------------------------------------------------------
# GPU 探针（与 setup_qflex.sh 同源，保证判据一致）
# 唯一判据 = 末尾的 PROBE_RESULT: OK（GPU 列得出来 ≠ 能算）
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
run_probe()     { [ -f "$PROBE_PY" ] || write_probe; "$1/bin/python" "$PROBE_PY" 2>&1; }
probe_ver()     { printf '%s\n' "$1" | sed -n 's/^jax_version: *//p' | tail -1; }
probe_ok()      { printf '%s\n' "$1" | grep -qE "^PROBE_RESULT: *OK"; }
is_rejected()   { [ -f "$JAX_REJECTED" ] && grep -qxF "$1" "$JAX_REJECTED"; }
reject_spec()   { is_rejected "$1" || printf '%s\n' "$1" >> "$JAX_REJECTED"; }

# ---------------------------------------------------------------------
# 1) 补齐 3 个 submodule（检出作者锁定的 commit）
clone_at() {   # $1=仓库内相对路径  $2=url  $3=期望 sha
  local rel="$1" url="$2" sha="$3" full="$QROOT/$1"
  if [ -d "$full/.git" ]; then
    local cur; cur="$(git -C "$full" rev-parse HEAD 2>/dev/null)"
    if [ "$cur" = "$sha" ]; then ok "$rel 已在期望 commit"; return 0; fi
    warn "$rel 当前 HEAD=$cur（≠ 期望 ${sha:0:12}）→ 重新 fetch"
  else
    if [ -d "$full" ] && [ -n "$(ls -A "$full" 2>/dev/null)" ]; then
      warn "$rel 非空且不是 git 仓库，内容: $(ls -A "$full" | head -3 | tr '\n' ' ')"
      info "  → 就地 git init（不删任何文件）；若 checkout 冲突会在这里报错"
    fi
    mkdir -p "$full"
    git init -q "$full"
    git -C "$full" remote add origin "$url" 2>/dev/null \
      || git -C "$full" remote set-url origin "$url"
  fi
  if ! git -C "$full" fetch --depth 1 origin "$sha" 2>/dev/null; then
    warn "浅 fetch 指定 commit 失败 → 改全量 fetch（可能较慢）"
    git -C "$full" fetch origin "$sha" || { no "$rel fetch 失败（网络？）"; return 1; }
  fi
  git -C "$full" checkout -q FETCH_HEAD || { no "$rel checkout 失败"; return 1; }
  local got; got="$(git -C "$full" rev-parse HEAD)"
  if [ "$got" = "$sha" ]; then
    ok "$rel @ ${sha:0:12}  ($(du -sh "$full" 2>/dev/null | cut -f1))"
  else
    no "$rel commit 不符: $got"
    return 1
  fi
}

if [ "$MODE" = "all" ] || [ "$MODE" = "submodules" ]; then
  say "1) 补齐 3 个 git submodule（检出作者锁定的 commit）"
  clone_at "envs/humanoid-bench"                "$SUBM_HB_URL" "$SUBM_HB_SHA"
  clone_at "envs/ostrich/ostrichrl"             "$SUBM_OS_URL" "$SUBM_OS_SHA"
  clone_at "envs/ms_700_walk/MS-Human-700"      "$SUBM_MS_URL" "$SUBM_MS_SHA"

  say "1.5) 键资源自检（代码里写死的路径，缺了环境起不来）"
  for f in \
    "envs/humanoid-bench/setup.py" \
    "envs/ostrich/ostrichrl/ostrichrl/assets/models/ostrich/ostrich.xml" \
    "envs/ms_700_walk/MS-Human-700/MS-Human-700.xml" \
    "envs/ms_700_walk/motion_data/walking_qpos.csv" ; do
    if [ -e "$QROOT/$f" ]; then ok "$f"; else no "$f   ← 缺失"; fi
  done
  info "依据: OstrichRun_v1.py:41 / locomotion_v1.py:43,232 / relax_humanoidbench.txt:3"
fi

# ---------------------------------------------------------------------
# 2) 建 3 套 venv
declare -A VENV_REQ VENV_DESC
VENV_REQ[relax_humanoidbench]="requirements/uv/relax_humanoidbench.txt"
VENV_REQ[relax_smplhumanoid]="requirements/uv/relax_smplhumanoid.txt"
VENV_REQ[relax_ms]="requirements/uv/relax_ms_700_walk.txt"
VENV_DESC[relax_humanoidbench]="h1-run-v0 / h1-balance_simple-v0"
VENV_DESC[relax_smplhumanoid]="SMPLHumanoidJump-v1"
VENV_DESC[relax_ms]="OstrichRun-v1 / MS700Locomotion-v1"
VENV_ORDER=(relax_humanoidbench relax_ms relax_smplhumanoid)   # SMPL 放最后（要拉 git 包，最不确定）

build_venv() {   # $1 = venv 名
  local NAME="$1" V="$QROOT/.venv/$1" REQ="$QROOT/${VENV_REQ[$1]}"
  say "2.$NAME  环境: ${VENV_DESC[$1]}"
  if [ ! -f "$REQ" ]; then no "找不到 $REQ"; return 1; fi

  # --- 2.1 创建 venv ---
  if [ -x "$V/bin/python" ]; then
    ok "venv 已存在: $("$V/bin/python" -V 2>&1)"
  else
    uv venv "$V" --python "$PYVER" || { no "uv venv 失败"; return 1; }
    ok "venv 已创建: $V"
  fi

  # --- 2.2 选定 jax spec：优先沿用已实测通过的，再依次试候选 ---
  local CANDS=()
  if [ "${FORCE_JAX_PROBE:-0}" != 1 ] && [ -s "$JAX_RECORD" ]; then
    CANDS+=("$(head -n1 "$JAX_RECORD" | tr -d '[:space:]')")
  fi
  CANDS+=("jax[cuda12]==0.4.38" "jax[cuda12]==0.5.3" "jax[cuda12]==0.6.2" \
          "jax[cuda12]==0.7.2" "jax[cuda12]==0.8.3")

  local SPEC="" OUT="" VER=""
  for S in "${CANDS[@]}"; do
    [ -z "$S" ] && continue
    # 已在本机实测失败的直接跳过（0.4.27 若被记过就在这里被跳过）
    if [ "${FORCE_JAX_PROBE:-0}" != 1 ] && is_rejected "$S"; then
      warn "跳过 $S（记录在 $JAX_REJECTED：本机实测探针失败）"; continue
    fi
    # 若该 venv 里已经是这个版本且探针通过，不必重装
    if [ -x "$V/bin/python" ]; then
      OUT="$(run_probe "$V")"; VER="$(probe_ver "$OUT")"
      if probe_ok "$OUT" && [ "jax==$VER" = "$S" -o "jax[cuda12]==$VER" = "$S" ]; then
        SPEC="$S"; ok "已是 $S 且 GPU 通过（跳过安装）"; break
      fi
    fi
    echo "   ---- 安装: $S ----"
    if uv pip install --python "$V" "$S"; then
      OUT="$(run_probe "$V")"; printf '%s\n' "$OUT" | sed 's/^/      /'
      if probe_ok "$OUT"; then SPEC="$S"; ok "GPU 可用，采用 $S"; break; fi
      no "$S 装上了但探针 FAIL"; reject_spec "$S"
    else
      no "$S 安装失败"
    fi
  done
  if [ -z "$SPEC" ]; then
    no "jax 候选全失败 —— 停在这里，把上面的输出发我，别继续装依赖"
    info "候选表: ${CANDS[*]}"
    return 1
  fi

  # --- 2.3 装其余依赖（★ 必须把 jax spec 一起交给 resolver，否则 jax 会被顶高）---
  # ★ v3 修正 A：h1 组必须加 --torch-backend=cpu，否则整个 venv 的依赖装不上。
  #   证据链：
  #     envs/humanoid-bench/setup.py:18      硬钉 torch==2.3.1
  #     PyPI torch 2.3.1 (linux/x86_64)      要求 nvidia-cudnn-cu12==8.9.2.26
  #     PyPI jax-cuda12-plugin 0.4.38        with-cuda extra 要求 nvidia-cudnn-cu12>=9.1,<10
  #   8.9.2.26 与 >=9.1 互斥 → uv 报 unsatisfiable。
  #   注意 jax-cuda12-plugin 0.4.27 的元数据里没有 with-cuda 的 nvidia 约束
  #   ⇒ 冲突是"为 Blackwell 把 jax 从 0.4.27 升到 0.4.38"引入的，不是原作者环境坏了。
  #   走 CPU 索引 → torch 变 2.3.1+cpu（不拉任何 nvidia-* 包），冲突消失，jax 不受影响。
  #   安全性：envs/humanoid-bench/humanoid_bench/env.py 不 import torch
  #   （torch 只被 humanoid-bench 自带的 baseline 算法用，relax 训练走 jax）。
  local TB=()
  [ "$NAME" = "relax_humanoidbench" ] && TB=(--torch-backend=cpu)
  echo ""
  info "装依赖: -r ${VENV_REQ[$1]}   （同时钉住 $SPEC）$([ ${#TB[@]} -gt 0 ] && echo '  + --torch-backend=cpu')"
  if uv pip install --python "$V" "${TB[@]}" "$SPEC" -r "$REQ"; then
    ok "依赖安装完成（一次过）"
  else
    warn "失败（多半是 src/futex.c 的 GCC 诊断）→ 回退放宽诊断等级，红色报错可忽略"
    if CFLAGS="$GCC_RELAX" uv pip install --python "$V" "${TB[@]}" "$SPEC" -r "$REQ"; then
      ok "依赖安装完成（CFLAGS 回退）"
    else
      warn "再回退: 全警告抑制"
      CFLAGS="-O2 -fPIC -w" uv pip install --python "$V" "${TB[@]}" "$SPEC" -r "$REQ" \
        || { no "依赖安装失败（$NAME）"; return 1; }
      ok "依赖安装完成（-w）"
    fi
  fi

  # --- 2.4 复验：装依赖有没有偷偷换掉 jax ---
  OUT="$(run_probe "$V")"; local POST; POST="$(probe_ver "$OUT")"
  if probe_ok "$OUT" && [ "jax[cuda12]==$POST" = "$SPEC" -o "jax==$POST" = "$SPEC" ]; then
    ok "复验通过：jax 仍是 $POST 且 GPU 可用"
  else
    warn "复验不符（选定 $SPEC / 实到 ${POST:-读不到}）→ 自动回滚"
    uv pip install --python "$V" "$SPEC" >/dev/null 2>&1
    OUT="$(run_probe "$V")"
    if probe_ok "$OUT"; then ok "回滚后 GPU 通过（jax $(probe_ver "$OUT")）"
    else no "回滚也没修好 —— 把输出发我"; return 1; fi
  fi
  return 0
}

if [ "$MODE" = "all" ] || [ "$MODE" = "envs" ]; then
  say "2) 建立其余环境组的 venv（当前只有 relax_myosuite）"
  info "已存在: $(ls "$QROOT/.venv" 2>/dev/null | tr '\n' ' ')"
  for NAME in "${VENV_ORDER[@]}"; do
    [ -n "$ONLY" ] && [ "$ONLY" != "$NAME" ] && continue
    build_venv "$NAME" || warn "$NAME 未完成，继续下一个"
  done
fi

# ---------------------------------------------------------------------
# 3) 验证
if [ "$MODE" = "all" ] || [ "$MODE" = "verify" ]; then
  say "3) 验证：各环境能否 import + gym.make（维度对论文 Figure 2(a) 自检）"
  info "预期 |S|   : $EXPECT_S"
  info "预期 |A|   : $EXPECT_A"

  check_env() {  # $1=venv  $2=环境id
    local V="$QROOT/.venv/$1" NAME="$2"
    [ -x "$V/bin/python" ] || { warn "跳过 $NAME（$1 不存在）"; return; }
    "$V/bin/python" - "$NAME" "$EXPECT_S" "$EXPECT_A" <<'PYEOF'
import sys, warnings, traceback
warnings.filterwarnings("ignore")
name, exp_s, exp_a = sys.argv[1], sys.argv[2], sys.argv[3]
def look(tbl, k):
    for tok in tbl.split():
        i, _, v = tok.partition(":")
        if i == k: return int(v)
    return None
try:
    # ★ v3 修正 B：原来只 import envs，漏了两个注册来源，导致
    #   myoLegWalk-v0 / myoHandPenTwirlRandom-v0 报 NameNotFound（假 FAIL）。
    #   证据：relax/env/__init__.py:8 `import myosuite`；
    #         myosuite/utils/__init__.py:22 `gym = import_gym()`（返回 gymnasium），
    #         由 myosuite/__init__.py 把 myo*-v0 注册进 gymnasium 注册表；
    #         envs/humanoid-bench/humanoid_bench/__init__.py 注册 h1-<task>-v0。
    try:
        import envs  # Qflex 自己的注册表（envs/__init__.py）
    except Exception as _e:
        print(f"  (warn) import envs: {type(_e).__name__}: {_e}")
    for _m in ("myosuite", "humanoid_bench"):
        try:
            __import__(_m)
        except Exception as _e:
            print(f"  (warn) import {_m}: {type(_e).__name__}: {_e}")
    import gymnasium as gym
    try:
        e = gym.make(name, render_mode="rgb_array")   # relax 的第一选择
    except Exception:
        e = gym.make(name)                             # relax 的回退
    s = e.observation_space.shape[0]; a = e.action_space.shape[0]
    es, ea = look(exp_s, name), look(exp_a, name)
    tag = "MATCH" if (es is None or (s == es and a == ea)) else f"MISMATCH (论文 {es}/{ea})"
    print(f"  [OK]   {name:26s} |S|={s:5d} |A|={a:4d}   {tag}")
    e.close()
except Exception:
    print(f"  [FAIL] {name}"); traceback.print_exc()
PYEOF
  }

  for e in h1-run-v0 h1-balance_simple-v0; do check_env relax_humanoidbench "$e"; done
  for e in OstrichRun-v1 MS700Locomotion-v1;        do check_env relax_ms            "$e"; done
  check_env relax_smplhumanoid SMPLHumanoidJump-v1
  # 顺带确认现有 venv 也能跑 myoHand（新的一格，零新环境成本）
  check_env relax_myosuite     myoHandPenTwirlRandom-v0
  check_env relax_myosuite     myoLegWalk-v0
fi

# ---------------------------------------------------------------------
say "汇总"
echo "  QROOT          : $QROOT"
echo "  venv 列表      : $(ls "$QROOT/.venv" 2>/dev/null | tr '\n' ' ')"
for d in envs/humanoid-bench envs/ostrich/ostrichrl envs/ms_700_walk/MS-Human-700; do
  if [ -d "$QROOT/$d/.git" ]; then
    echo "  $d @ $(git -C "$QROOT/$d" rev-parse --short HEAD)"
  else
    echo "  $d   ★未初始化"
  fi
done
echo "  API 补丁(共享) : $(grep -c 'FIX-1\|FIX-2' "$QROOT/relax/utils/persistence.py" 2>/dev/null) 个标记（应为 2；补丁在源码里，所有 venv 共享）"
echo "  日志           : ~/setup_other_envs.log"
echo ""
ok "下一步（第 2 步冒烟）—— ★ 必须带 --save_policy_every 300000："
info "  否则 save_policy_every 默认 = min(1e6, total_step/20) = 15000，300000 步会触发"
info "  **20 次评估**（每次 5 局、独立 CPU 子进程），评估时间会盖过训练时间，"
info "  测出来的 wall-clock 就不是训练速率了。"
cat <<'CMDS'
    cd ~/work/Qflex
    export XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false

    # 每个环境只冒烟 1 个算法（qflex），300000 步
    .venv/relax_humanoidbench/bin/python scripts/train.py --alg qflex --env h1-run-v0            --seed 100 --total_step 300000 --num_vec_envs 70 --save_policy_every 300000 --suffix smoke
    .venv/relax_humanoidbench/bin/python scripts/train.py --alg qflex --env h1-balance_simple-v0  --seed 100 --total_step 300000 --num_vec_envs 70 --save_policy_every 300000 --suffix smoke
    .venv/relax_ms/bin/python            scripts/train.py --alg qflex --env OstrichRun-v1        --seed 100 --total_step 300000 --num_vec_envs 80 --save_policy_every 300000 --suffix smoke
    .venv/relax_smplhumanoid/bin/python  scripts/train.py --alg qflex --env SMPLHumanoidJump-v1  --seed 100 --total_step 300000 --num_vec_envs 80 --save_policy_every 300000 --suffix smoke
    .venv/relax_myosuite/bin/python      scripts/train.py --alg qflex --env myoHandPenTwirlRandom-v0 --seed 100 --total_step 300000 --num_vec_envs 80 --save_policy_every 300000 --suffix smoke

    # MS700 单独对待：官方配置 224 并行环境 / 5000 万步，本机 12G 显存大概率放不下
    .venv/relax_ms/bin/python            scripts/train.py --alg qflex --env MS700Locomotion-v1  --seed 100 --total_step 300000 --num_vec_envs 16 --save_policy_every 300000 --suffix smoke_ms700
CMDS
