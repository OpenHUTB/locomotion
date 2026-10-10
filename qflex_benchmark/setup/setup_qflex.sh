#!/usr/bin/env bash
# =====================================================================
# Qflex (relax_myosuite) 环境搭建 —— WSL / Ubuntu 一键脚本  v8
#
# ★ 阅读提示：下方"vN 相对 vN-1 的修正"是本脚本的开发迭代记录，每条都附当时的实测依据
#   （属于背景资料，可按需跳读）。本仓库只含最终版 —— 即上面这个版本号 —— 历史版本不在仓库内。
#
# v8 相对 v7 的修正（依据 = 2026-09-24 00:37 完整实跑日志）:
#   [1] ★★ 实锤（非推断）：官方 requirements 锁定的 jax 0.4.27 **在本机
#       （RTX 5070 Ti Laptop / sm_120）上无法执行任何 GPU 算子**。探针原文：
#         XlaRuntimeError: INTERNAL: ptxas exited with non-zero error code 65280,
#         ptxas fatal : Program with .target 'sm_90a' cannot be compiled to
#         future architecture
#       成因：jaxlib 0.4.27 自带的 ptxas 只认到 sm_90a，无法为更新的
#       Blackwell 架构编译 —— 是**硬不可用**，不是"版本老所以慢"。
#       同一探针下 0.4.38 报 MATMUL_OK: 1073741824.0（=1024³，值正确）。
#       → 本机只有一条路：jax >= 0.4.38 + persistence.py 兼容补丁（3.5 步）。
#   [2] ★ v7 把 0.4.27 放候选首位的代价被实测暴露：试 0.4.27 要额外解析 +
#       下载 nvidia-cudnn-cu12 8.9.7（672.1MiB），实测 "Prepared 5 packages
#       in 24m 41s"。每次重跑本脚本都会白等这 20+ 分钟。
#       → 新增 ①「2.5 快速通道」：记录文件 .jax_version_ok 里的 spec 若与
#            当前环境实测版本一致且探针通过，直接沿用，跳过整轮候选循环；
#          ②「.jax_rejected 失败缓存」：探针实测失败的 spec 记下来，重跑时
#            直接跳过（FORCE_JAX_PROBE=1 可强制重试；换到非 Blackwell 机器
#            时删掉该文件，0.4.27 就会重新参与探测）。
#   [3] ★ 冒烟提示的步数 20000 → 100000，并补两条经源码核对的口径：
#       ① evaluator 子进程按 save_policy_every = min(1e6, total_step/20)
#          评估 5 局（off_policy.py:302 / :87）→ **评估次数恒为 20 次**，
#          total_step 越小它占比越大，短冒烟的总时长可能由它主导；
#       ② 该子进程写出 <log_path>/log.csv（step,avg_ret,std_ret）——
#          这就是学习曲线的原始数据（不是 tensorboard 标量）。
#   [4] 静态核查（从 PyPI 下 jax 0.4.38 wheel 逐符号核对，非推测）：
#       persistence.py 在 0.4.38 下的**唯一**不兼容点就是 named_shape 那两行。
#       已确认存在：jax.core.eval_jaxpr(core.py:54)、
#       jax.make_jaxpr(..., return_shape=True)(_src/api.py:2070)、
#       jax.tree.flatten(tree.py)、jax.core.Primitive / ClosedJaxpr / Jaxpr
#       （仅 DeprecationWarning，返回值可用）、jaxpr.effects、
#       jaxlib.xla_client.Traceback（jax/lib/xla_client.py:27 硬引用，
#       且 jax/__init__.py:243 `import jax.lib` 保证它一定会被加载）。
#       另记一处死雷（不影响本流程，不改作者源码）：flow_exp.py:434 用了
#       jnp.random.key，而 jax.numpy.random 在 0.4.38 已不存在 ——
#       该 override 无人调用。
#
# v7 相对 v6 的修正（依据 = 2026-09-23 23:55 训练冒烟崩溃日志）:
#   [1] ★★ 冒烟跑到 trainer.setup() 崩在
#         AttributeError: 'ShapeDtypeStruct' object has no attribute 'named_shape'
#       根因（从 PyPI 下 wheel 读 jax/_src/api.py 实测，非推测）：
#         jax 0.4.27  __slots__ = [shape,dtype,named_shape,sharding,_dll]   → 有
#         jax 0.4.35  __slots__ = [shape,dtype,sharding,_dll,weak_type]     → 已移除
#       而 relax/utils/persistence.py:39-40 无条件访问该属性 → 必崩。
#       → ① JAX 候选把官方锁定的 **0.4.27** 放到首位（它同时满足
#            "官方验证过的 API" 和 "有 named_shape"）；
#          ② 探针增加 NAMED_SHAPE_OK 诊断行；
#          ③ 新增 3.5 步 API 兼容检查：若选中的版本缺 named_shape，
#            自动为 persistence.py:39-40 打等价补丁（getattr 默认 {}），
#            幂等 + 留 .orig 备份 + 只在那一处改动，不碰算法。
#

# 用法（在 qflex_benchmark/ 目录下执行；路径可用 QROOT 覆盖）:
#   QROOT=~/work/Qflex bash setup/setup_qflex.sh 2>&1 | tee ~/setup_qflex.log
#   （仓库内文件已是 LF 行尾，直接跑即可；仅当你从 Windows 手动拷到 WSL 时才需要：
#      tr -d '\r' < setup_qflex.sh > ~/setup_qflex.sh）
#
# 说明:
#   * 幂等，可重复运行；已完成的步骤会自动跳过
#   * 全程不需要 sudo（系统库请在脚本外单独 apt 安装）
#   * 分段执行:  bash ~/setup_qflex.sh jax | deps | smoke
#   * 覆盖 Python 版本:  PYVER=3.12 bash ~/setup_qflex.sh
#   * 手动指定 PyPI 镜像:  IDX=https://mirrors.aliyun.com/pypi/simple bash ~/setup_qflex.sh
#
# v6 相对 v5 的修正（依据 = 2026-09-23 23:35 用户手敲命令失败）:
#   [1] ★ uv 的 PATH 未持久化 —— v5 只在脚本进程里 export
#       PATH="$HOME/.local/bin:$PATH"，脚本一结束就没了；用户在**新终端**
#       手敲 `uv ...` 直接 "Command 'uv' not found"（实测踩到，用户还误试了
#       snap 版）。→ 第 1 步末尾把 ~/.local/bin 写进 ~/.bashrc（幂等，
#       grep 命中就不重复追加）。
#   [2] 文档化的两条禁令写进脚本注释：① 不要 sudo 装 uv/跑 uv（会把项目
#       文件变成 root 属主，后续训练写 logs 报 Permission denied）；
#       ② 不要用 snap 版 uv 覆盖（classic confinement + 与已有二进制冲突）。
#
# v5 相对 v4 的修正（依据 = 2026-09-23 23:25 实跑日志）:
#   [1] ★ 第 3 步在本机 **成功**：jax 0.4.38 实测
#       devices=['cuda:0']、DEVICE_KIND='NVIDIA GeForce RTX 5070 Ti Laptop GPU'、
#       MATMUL_OK=1073741824.0（=1024³，数值正确）。
#       → 推翻"0.4.38 跑不了 Blackwell"的预判，也用不着升到 0.6.x；
#         保持最接近官方 0.4.27 的版本 = API 兼容风险最小。
#       机制：jax-cuda12-plugin 0.4.38 对 nvidia-* 只给**下限**(>=12.1.105)，
#       解析器装的是最新 nvidia-cuda-*==12.9.x + cudnn==9.26 → sm_120 有支持。
#   [2] ★ 真正卡住的是第 4 步编译 src/futex.c：Ubuntu 26.04 的 GCC（GCC 14+）
#       把 -Wincompatible-pointer-types 从 warning 提升为 **error**。
#       报错点 4 处（futex.c:65/83/98/108）：函数签名收 struct lock_t*，
#       实际传的是 atomic_uint*；逻辑本身没错（函数体立刻用 LOCK_COUNT(lock)
#       按结构体布局取回字段），只是签名类型写歪了。
#       → v4 的回退 CFLAGS='-std=gnu11' **无效**（这是类型诊断，不是 C 标准
#         版本问题）。回退链换成 -Wno-error -Wno-incompatible-pointer-types，
#         再用 -w 全抑制兜底。★ 不改作者源码，只放编译器诊断等级。
#   [3] 汇总块增加"C扩展编译方式"一行（便于判断走的是哪条回退）。
#
# v4 相对 v3 的修正（依据 = 2026-09-23 23:25 对探针判据的复核）:
#   [1] ★ 探针判据原来只看 GPU_COUNT，**会误判**：jax.devices() 能列出 GPU
#       不等于能算 —— 在 sm_120 上用不支持它的工具链时，惯常表现就是
#       "设备列得出来，但 XLA 编译/执行抛错"。而 GPU_COUNT 在探针里**先于**
#       matmul 打印（旧版 220 行 vs 223-226 行），所以 matmul 崩了它照样
#       输出 1 → 旧判据返回真、脚本提前停在不能算的版本上。
#       → 探针末尾统一打印 PROBE_RESULT: OK|FAIL 作为**唯一**判据；
#         GPU_COUNT 降级为诊断信息；matmul 结果过 math.isfinite() 排除 nan/inf。
#   [2] probe_has_gpu() 改为 grep "^PROBE_RESULT: *OK"，第 4.5 步复验自动同严。
#
# v3 相对 v2 的修正（依据 = 2026-09-23 23:06 实跑日志 + 8 个镜像实测）:
#   [1] ★ 镜像必须探测后再用 —— 实测 `pypi.tuna.tsinghua.edu.cn` 与
#       `mirrors.bfsu.edu.cn` 对客户端的 `/simple/jax/` 返回 **403 Forbidden**，
#       导致 v2 第 3 步 5 个 JAX 候选全部"安装失败"（uv 的 hint 里写明是 403）。
#       实测可用: aliyun / ustc / nju / tencent-cloud / huaweicloud / sjtu。
#       → 新增「0.5 步」逐源探测（curl `/jax/` 要 200 且含 .whl），
#         并把第 3 步改成"一个源试完全部候选，失败自动换下一个源"。
#   [2] ~/.bashrc 里上一次写死的 tuna 配置会被重写清掉，否则新终端又 403。
#   [3] 汇总块改为调用 /tmp/_qflex_versions.py 输出全部关键包版本
#       （v2 那行 numpyro/optax 因为引号转义 bug 报 SyntaxError）。
#
# v2 相对 v1 的修正（依据 = PyPI 元数据 + pip 解析器实测，不是推测）:
#   [1] 第 4 步确实会动 jax —— 实测 optax 最新版要 jax>=0.5.3、numpyro 最新版
#       要 jax>=0.7.0，而 base.txt 里两个都没锁版本。「什么都不钉」时 pip 把
#       jax 从 0.4.38 顶到了 0.7.0（真实 Linux 环境只会更高）。
#       → 第 4 步改为把第 3 步选定的 jax spec 一起交给 resolver。实测这样
#         jax 保持 0.4.38 不动，resolver 自动回溯出 numpyro 0.19.0 +
#         optax 0.2.5（与 0.4.38 兼容的上界），19 个包解析成功。
#   [2] 新增「4.5 复验」: 装完依赖重跑 GPU 探针，与记录版本比对；
#       不一致就自动回滚 jax 并再次复验。v1 完全没有这一步。
#   [3] 冒烟补测完整 import 链: relax.env → relax.env.vector →
#       process3 → relax.futex，并真起一个 2 进程 futex 向量环境走几步。
#       （v1 只单独 import 了 relax.futex，等于没测这条链）
#   [4] JAX 候选列表加上上界: 不钉版本号的话 jax 会解析到远新于官方 0.4.27
#       的版本（实测至少 0.7.0；PyPI 最新已到 0.11.2），代码大概率跑不动。
#       → 逐个钉版本到 0.8.3，只有全失败才用无版本 cuda13 兜底并告警。
#   [5] pip 回退路径补一条「不需要 pip」的兜底: 从 GitHub 镜像直接取
#       静态 uv 二进制（v1 只 warn，等于断了）。
# =====================================================================

MODE="${1:-all}"                 # all | jax | deps | smoke
QROOT="${QROOT:-$HOME/work/Qflex}"
PYVER="${PYVER:-3.12}"
VENV="$QROOT/.venv/relax_myosuite"
JAX_RECORD="$QROOT/.jax_version_ok"
JAX_REJECTED="$QROOT/.jax_rejected"   # 本机探针实测失败的 spec（重跑时直接跳过）
PROBE_PY="/tmp/_jaxprobe.py"

CH=$'\033[1;36m'; CO=$'\033[1;32m'; CN=$'\033[1;31m'; CY=$'\033[1;33m'; CE=$'\033[0m'
say() { printf '\n%s%s===== %s =====%s\n' "$CH" "" "$*" "$CE"; }
ok()  { printf '%s[OK]%s   %s\n' "$CO" "$CE" "$*"; }
no()  { printf '%s[FAIL]%s %s\n' "$CN" "$CE" "$*"; }
warn(){ printf '%s[WARN]%s %s\n' "$CY" "$CE" "$*"; }
info(){ printf '       %s\n' "$*"; }

# --- PyPI 镜像探测（国内若干源会对非浏览器客户端返回 403，必须实测再选）---
IDX_OK=()
probe_index() {
  local u="$1" code
  code="$(curl -s -o /tmp/_idx.html -w '%{http_code}' --max-time 15 "$u/jax/" 2>/dev/null)"
  [ "$code" = "200" ] || return 1
  grep -q '\.whl' /tmp/_idx.html 2>/dev/null
}

GPU_OK=0
UV_OK=0
JAX_USED=""        # 完整 spec, 例如 jax[cuda12]==0.4.38
JAX_USED_VER=""    # 只版本号, 例如 0.4.38
JAX_FINAL_OK=0     # 第 4 步之后 jax 仍然正常
JAX_ROLLBACK=""
JAX_UNPINNED=0     # 兜底分支才置 1（spec 无版本约束）
JAX_FAST=0         # 2.5 快速通道命中（沿用记录，跳过候选循环）
CFLAGS_USED="${CFLAGS:-}"   # 第 4 步实际生效的编译参数
                            # 空 = setuptools 默认、未走回退；
                            # 外部预先 export CFLAGS 时直接取用（汇总才准确）
# 放宽 GCC 诊断等级的编译参数（实测有效：futex.c 的 4 处指针类型不匹配在
# GCC 14+ 是 error，加 -Wno-error 后按 warning 处理，不改源码）
GCC_RELAX_DOC="-O2 -fPIC -DNDEBUG -std=gnu11 -Wno-error -Wno-incompatible-pointer-types"

# ---------------------------------------------------------------------
say "0) 配置 uv 缓存与 Python 下载镜像"
export UV_PYTHON_INSTALL_MIRROR="https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/"
export UV_CACHE_DIR="$QROOT/.uv-cache"
export UV_NO_PROGRESS=1
mkdir -p "$UV_CACHE_DIR"
info "UV_PYTHON_INSTALL_MIRROR = $UV_PYTHON_INSTALL_MIRROR"
info "UV_CACHE_DIR             = $UV_CACHE_DIR"

# ---------------------------------------------------------------------
say "0.5) 探测可用的 PyPI 镜像（tuna / bfsu 目前对客户端返回 403，不能硬编码）"
MIRROR_CANDIDATES=(
  "https://mirrors.aliyun.com/pypi/simple"
  "https://mirrors.ustc.edu.cn/pypi/simple"
  "https://mirrors.nju.edu.cn/pypi/web/simple"
  "https://mirrors.cloud.tencent.com/pypi/simple"
  "https://repo.huaweicloud.com/repository/pypi/simple"
  "https://mirror.sjtu.edu.cn/pypi/web/simple"
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://pypi.org/simple"
)
if [ -n "${IDX:-}" ]; then
  info "使用手动指定的索引: $IDX"
  IDX_OK=("$IDX")
else
  for U in "${MIRROR_CANDIDATES[@]}"; do
    if probe_index "$U"; then
      ok   "可用: $U"
      IDX_OK+=("$U")
    else
      warn "不可用（403 / 超时 / 无内容）: $U"
    fi
  done
fi
if [ "${#IDX_OK[@]}" = 0 ]; then
  no "没有任何可用镜像 —— 终止。可手动指定: IDX=<你的镜像> bash ~/setup_qflex.sh"
  exit 1
fi
export UV_DEFAULT_INDEX="${IDX_OK[0]}"
ok "选定索引: $UV_DEFAULT_INDEX（共 ${#IDX_OK[@]} 个可用，第 3 步失败时会自动换下一个）"

# 重写 ~/.bashrc 里的镜像块（上次失败的 tuna 配置要清掉，否则新终端又会 403）
if [ -f "$HOME/.bashrc" ]; then
  sed -i '/^# --- Qflex \/ uv 国内镜像/d; /^export UV_DEFAULT_INDEX=/d; /^export UV_PYTHON_INSTALL_MIRROR=/d' "$HOME/.bashrc"
fi
cat >> "$HOME/.bashrc" <<EOF
# --- Qflex / uv 国内镜像 (由 setup_qflex.sh 写入) ---
export UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX"
export UV_PYTHON_INSTALL_MIRROR="$UV_PYTHON_INSTALL_MIRROR"
EOF
ok "已刷新 ~/.bashrc（新开终端永久生效）"

# ---------------------------------------------------------------------
say "1) 安装 uv"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  ok "已有 uv: $(uv --version)"
  UV_OK=1
else
  echo "-- 方式1: 官方安装脚本 (astral.sh)"
  curl -LsSf --max-time 180 https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null 2>&1 && { ok "uv 安装成功: $(uv --version)"; UV_OK=1; }

  if [ "$UV_OK" != 1 ]; then
    echo "-- 方式2: 回退到清华 PyPI 用 pip 安装"
    if command -v pip3 >/dev/null 2>&1; then
      pip3 install --user --break-system-packages -q \
        -i "${IDX_OK[0]:-https://mirrors.aliyun.com/pypi/simple}" uv
      command -v uv >/dev/null 2>&1 && { ok "uv(pip) $(uv --version)"; UV_OK=1; }
    else
      warn "系统没有 pip3（Ubuntu 默认不装 python3-pip）"
      info "若想用这条路径: sudo apt install -y python3-pip  (需要 sudo，脚本不代跑)"
    fi
  fi

  if [ "$UV_OK" != 1 ]; then
    echo "-- 方式3: 不依赖 pip —— 直接从 GitHub 镜像取静态 uv 二进制"
    for U in \
      "https://ghfast.top/https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" \
      "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
    do
      echo "   试: $U"
      if curl -L --max-time 300 -o /tmp/_uv.tgz "$U" && tar -xzf /tmp/_uv.tgz -C /tmp 2>/dev/null; then
        BIN="$(find /tmp -maxdepth 3 -type f -name uv -perm -u+x 2>/dev/null | head -1)"
        if [ -n "$BIN" ]; then
          mkdir -p "$HOME/.local/bin" && cp "$BIN" "$HOME/.local/bin/uv"
          chmod +x "$HOME/.local/bin/uv"
          command -v uv >/dev/null 2>&1 && { ok "uv(静态二进制) $(uv --version)"; UV_OK=1; break; }
        fi
      fi
      warn "该源失败"
    done
  fi
fi
if [ "$UV_OK" != 1 ]; then
  no "uv 安装失败 —— 终止。请手动装好 uv 后重跑本脚本。"
  exit 1
fi

# ★ uv 落在 ~/.local/bin，而上面的 export 只作用于本脚本进程。
#   不持久化的话，用户在**新终端**手敲 `uv ...` 会 Command 'uv' not found。
#   幂等：任一行已含 .local/bin（astral 安装器可能已写过）就不重复追加。
#   注：不要 sudo 装 uv / 跑 uv —— 会把项目文件变成 root 属主，
#       之后训练写 logs/ 会报 Permission denied。
if [ -f "$HOME/.bashrc" ] && ! grep -q '\.local/bin' "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'EOF'

# --- uv 可执行文件目录 (由 setup_qflex.sh 写入) ---
export PATH="$HOME/.local/bin:$PATH"
EOF
  ok "已把 ~/.local/bin 写入 ~/.bashrc（新终端可直接用 uv）"
fi

# ---------------------------------------------------------------------
say "2) 准备 Python $PYVER 虚拟环境"
cd "$QROOT" 2>/dev/null || { no "找不到项目目录 $QROOT"; exit 1; }
info "项目目录: $(pwd)"

if [ -x "$VENV/bin/python" ]; then
  ok "venv 已存在: $VENV ($("$VENV/bin/python" -V 2>&1))"
else
  MIRRORS=(
    "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/"
    "https://mirror.lzu.edu.cn/github-release/astral-sh/python-build-standalone/"
    "https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download"
    "https://python-standalone.org/mirror/astral-sh/python-build-standalone"
  )
  PY_OK=0
  for M in "${MIRRORS[@]}"; do
    echo "-- 试镜像: $M"
    if UV_PYTHON_INSTALL_MIRROR="$M" uv python install "$PYVER"; then
      ok "Python $PYVER 下载成功（镜像: $M）"
      export UV_PYTHON_INSTALL_MIRROR="$M"
      PY_OK=1
      break
    fi
    warn "该镜像失败，换下一个"
  done
  [ "$PY_OK" = 1 ] || warn "所有镜像都失败，仍尝试用默认源创建 venv"
  uv venv "$VENV" --python "$PYVER" || { no "venv 创建失败"; exit 1; }
  ok "venv 创建完成: $VENV"
fi

# ---------------------------------------------------------------------
# GPU 探针脚本（第 3 步与第 4 步复验共用同一份，保证可比）
#
# 判据设计（v4 修正）：**只有能真正在 GPU 上完成一次矩阵乘才算通过**。
#   * GPU_COUNT 只是设备枚举结果，jax.devices() 能列出 GPU **不等于**能算：
#     在 sm_120 上用了不支持它的 CUDA 工具链时，惯常表现就是"设备列得出来，
#     但 XLA 编译/执行抛错"。v3 只看 GPU_COUNT，会在这里误判成功并提前停。
#   * 因此把 GPU_COUNT 降级为诊断信息，唯一判据是最后一行 PROBE_RESULT。
#   * matmul 结果必须过 math.isfinite() —— 排除 nan / inf 这类"跑通了但结果无效"。
write_probe() {
  cat > "$PROBE_PY" <<'PYEOF'
import os, math, traceback
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
ok = False
try:
    import jax, jax.numpy as jnp, numpy as np
    print("jax_version:", jax.__version__)
    print("numpy_version:", np.__version__)
    d = jax.devices()
    print("devices:", [str(x) for x in d])
    gpu = [x for x in d if x.platform == "gpu"]
    print("GPU_COUNT:", len(gpu))
    # Qflex 的 relax/utils/persistence.py:39-40 直接访问 ShapeDtypeStruct.named_shape，
    # 而该属性在 jax 0.4.35 起被移除（0.4.27 有 / 0.4.35+ 无）。
    # 只作诊断上报，不参与选版本判据 —— 缺了可以用兼容补丁兜，GPU 不可用则无解。
    print("NAMED_SHAPE_OK:", hasattr(jax.ShapeDtypeStruct((1,), "float32"), "named_shape"))
    if not gpu:
        print("NOTE: 未发现 gpu 设备")
    else:
        print("DEVICE_KIND:", gpu[0].device_kind)
        a = jnp.ones((1024, 1024), dtype=jnp.float32)
        s = (a @ a).sum()
        s.block_until_ready()
        v = float(s)
        if math.isfinite(v):
            print("MATMUL_OK:", v)
            ok = True
        else:
            print("MATMUL_BAD: 结果非有限值 ->", v)
except Exception as e:
    traceback.print_exc()
    print("PROBE_ERR:", type(e).__name__, str(e)[:300])
print("PROBE_RESULT:", "OK" if ok else "FAIL")
PYEOF
}
run_probe() { "$VENV/bin/python" "$PROBE_PY" 2>&1; }
probe_ver() { printf '%s\n' "$1" | sed -n 's/^jax_version: *//p' | tail -1; }
# 唯一判据：GPU 列得出来 且 matmul 真的在 GPU 上算出了有限值
probe_has_gpu() { printf '%s\n' "$1" | grep -qE "^PROBE_RESULT: *OK"; }

# ---------------------------------------------------------------------
if [ "$MODE" = "all" ] || [ "$MODE" = "jax" ]; then
  write_probe

  # -------------------------------------------------------------------
  # 2.5) 快速通道 —— 见文件头 v8 说明 [2]
  #   不再为了"顺手试一下 0.4.27"反复下载 672MiB 的 nvidia-cudnn-cu12 8.9.7
  #   （实测一次 24m41s）。命中条件（三条同时满足）：
  #     记录文件存在 / 记录版本 == 当前环境实测版本 / 探针 PROBE_RESULT: OK
  #   想强制重新探测全部候选：FORCE_JAX_PROBE=1 FORCE_JAX_PROBE 见下方 3 步。
  # -------------------------------------------------------------------
  say "2.5) 快速通道：复用上次验证通过的 jax（跳过已实测失败的候选）"
  if [ "${FORCE_JAX_PROBE:-0}" = 1 ]; then
    info "FORCE_JAX_PROBE=1 → 强制走完整候选循环"
  elif [ -f "$JAX_RECORD" ] && [ -x "$VENV/bin/python" ]; then
    REC_SPEC="$(head -n 1 "$JAX_RECORD" | tr -d '[:space:]')"
    REC_OUT="$(run_probe)"
    REC_VER="$(probe_ver "$REC_OUT")"
    REC_SPEC_VER="$(printf '%s' "$REC_SPEC" | sed -n 's/.*==//p')"
    info "记录: ${REC_SPEC:-（空）} | 实测 jax: ${REC_VER:-读不到} | $(printf '%s' "$REC_OUT" | grep -oE '^PROBE_RESULT: *[A-Z]+')"
    if probe_has_gpu "$REC_OUT" && [ -n "$REC_VER" ] && [ "$REC_VER" = "$REC_SPEC_VER" ]; then
      JAX_USED="$REC_SPEC"; JAX_USED_VER="$REC_VER"; GPU_OK=1; JAX_FAST=1
      ok "沿用 $REC_SPEC（实测 jax $REC_VER，GPU 复验通过）→ 跳过第 3 步候选循环"
    else
      warn "记录与实际环境不符 → 走完整候选循环（可能要重装/重下，20 分钟以上）"
    fi
  else
    info "尚无记录文件 $JAX_RECORD → 走完整候选循环"
  fi

  # ---- 完整候选循环（仅在快速通道未命中时执行）----
  if [ "$JAX_FAST" != 1 ]; then
  say "3) 安装 JAX(CUDA) 并逐版本验证能否识别 RTX 5070 Ti"

  # 候选列表：从小到大逐级试，找到第一个能在 sm_120 上出 GPU 的就停。
  # 全部钉住版本号 —— 不写死的话 "jax[cuda12]" 今天会解析到 0.11.x，
  # 跟官方锁的 0.4.27 差 7 个 minor，Qflex 代码大概率直接崩。
  JAX_SPECS=(
    "jax[cuda12]==0.4.27"   # ★ 官方 requirements/uv/jax-cuda12.txt 锁定的版本；
                            #   也是最后一个带 ShapeDtypeStruct.named_shape 的版本
                            #   （0.4.35 起该属性被移除 → persistence.py:39 会崩）
                            #   ★★ 但 2026-09-24 实测：本机（sm_120 / Blackwell）
                            #   上它连一个 GPU 算子都编译不了 ——
                            #     ptxas fatal : Program with .target 'sm_90a'
                            #     cannot be compiled to future architecture
                            #   （jaxlib 0.4.27 自带 ptxas 只认到 sm_90a，硬限制）
                            #   → 在本机会被 .jax_rejected 自动跳过；仍留在候选表里
                            #     是为了兼容非 Blackwell 机器 —— 那里它反而是首选，
                            #     因为它不需要 3.5 步的 API 兼容补丁。
    "jax[cuda12]==0.4.38"
    "jax[cuda12]==0.5.3"
    "jax[cuda12]==0.6.2"
    "jax[cuda12]==0.7.2"
    "jax[cuda12]==0.8.3"
  )

  # 本机实测失败的 spec 缓存（v8）：探针 PROBE_RESULT: FAIL 就记下来，重跑时
  # 直接跳过 —— 否则每次都要为必然失败的版本重新解析并下载 672MiB 的
  # nvidia-cudnn-cu12 8.9.7（实测 24m41s）。
  # ★ 换到非 Blackwell 机器（A100/H100 等）时删掉它，0.4.27 就会重新参与探测。
  is_rejected() { [ -f "$JAX_REJECTED" ] && grep -qxF "$1" "$JAX_REJECTED"; }
  reject_spec() { is_rejected "$1" || printf '%s\n' "$1" >> "$JAX_REJECTED"; }

  # 用当前 UV_DEFAULT_INDEX 把钉版候选试一遍；装成功且探针出 GPU 就返回 0
  try_specs() {
    local S PROBE_OUT
    for S in "${JAX_SPECS[@]}"; do
      if [ "${FORCE_JAX_PROBE:-0}" != 1 ] && is_rejected "$S"; then
        warn "跳过 $S —— 本机已实测探针失败（记录于 $JAX_REJECTED；要重试: FORCE_JAX_PROBE=1）"
        continue
      fi
      echo ""
      echo "---- 尝试: $S ----"
      if uv pip install --python "$VENV" "$S"; then
        PROBE_OUT="$(run_probe)"
        printf '%s\n' "$PROBE_OUT" | sed 's/^/      /'
        if probe_has_gpu "$PROBE_OUT"; then
          JAX_USED="$S"; GPU_OK=1
          JAX_USED_VER="$(probe_ver "$PROBE_OUT")"
          ok "GPU 可用！采用 $S (实测 jax $JAX_USED_VER)"
          return 0
        else
          no "$S 装上了但没识别到 GPU"
          reject_spec "$S"
        fi
      else
        no "$S 安装失败（依赖冲突或下载失败）"
      fi
    done
    return 1
  }

  # 逐个可用索引重试：某个源被限流/缺包时自动换下一个
  for IDXN in "${IDX_OK[@]}"; do
    export UV_DEFAULT_INDEX="$IDXN"
    if [ "${#IDX_OK[@]}" -gt 1 ]; then
      echo ""
      info ">>> 本轮使用索引: $IDXN"
    fi
    try_specs && break
  done

  # 兜底：无版本约束。会拉到当前最新，代码兼容性存疑，故告警。
  if [ "$GPU_OK" != 1 ]; then
    warn "钉版候选全部失败，最后尝试无版本约束的 jax[cuda13]（会拉到最新版）"
    warn "即使 GPU 探针通过，jax 版本也远新于官方 0.4.27，训练代码可能不兼容"
    for IDXN in "${IDX_OK[@]}"; do
      export UV_DEFAULT_INDEX="$IDXN"
      if uv pip install --python "$VENV" "jax[cuda13]" || uv pip install --python "$VENV" "jax[cuda12]"; then
        PROBE_OUT="$(run_probe)"
        printf '%s\n' "$PROBE_OUT" | sed 's/^/      /'
        if probe_has_gpu "$PROBE_OUT"; then
          JAX_USED="jax[cuda13]"; JAX_UNPINNED=1
          GPU_OK=1
          JAX_USED_VER="$(probe_ver "$PROBE_OUT")"
          ok "GPU 可用（jax $JAX_USED_VER，无版本约束，注意兼容性风险）"
          break
        fi
      fi
    done
  fi

  if [ "$GPU_OK" = 1 ]; then
    printf '%s\n' "$JAX_USED" > "$JAX_RECORD"
    ok "已记录可用 spec 到 $JAX_RECORD"
  else
    no "所有 JAX 版本都未能识别到 GPU —— 先别继续，把上面的输出发给我"
  fi
  fi   # ← 结束 "if [ "$JAX_FAST" != 1 ]"（完整候选循环）
  else
    # 分段模式（deps / smoke）下，从记录文件取回上次选定的 jax
    if [ -f "$JAX_RECORD" ]; then
      JAX_USED="$(cat "$JAX_RECORD")"
      JAX_USED_VER="$(printf '%s' "$JAX_USED" | sed -n 's/.*==//p')"
      info "复用已记录的 JAX spec: $JAX_USED"
    else
      warn "没有 $JAX_RECORD，第 4 步将无法钉住 jax 版本"
    fi
    # 记录里拿不到版本号（例如无约束 spec）→ 直接探针当前环境当基线
    if [ -z "$JAX_USED_VER" ] && [ -x "$VENV/bin/python" ]; then
      write_probe
      JAX_USED_VER="$(probe_ver "$(run_probe)")"
      info "记录里没有版本号，改用当前环境实测基线: ${JAX_USED_VER:-读不到}"
    fi
  fi

# ---------------------------------------------------------------------
# 3.5) 兼容补丁：relax/utils/persistence.py（jax >= 0.4.35 / 0.4.38 适配）
#
# 两处不兼容都是实测定界的，不是推测（详见下方内嵌补丁器头部注释）：
#   FIX-1 named_shape : 0.4.35 起 ShapeDtypeStruct.named_shape 被移除
#         → 训练在 OffPolicyTrainer.setup() 崩:
#           AttributeError: 'ShapeDtypeStruct' object has no attribute 'named_shape'
#   FIX-2 config State: jax._src.config.State（79 个配置 flag，基类是 pybind11 的
#         xla_extension.config.Config，tp_new 缺失）无法 pickle；它以「绑定方法」
#         形式出现在导出的 jaxpr 里（State.__call__ ← JaxprEqnContext /
#         jax._src.compute_on.extend_compute_type）→ 崩在:
#           TypeError: cannot pickle 'State' object
#         修法：按名持久化（"S:<flag 属性名>"），读回时取当前进程的同名活对象；
#         绝不能丢成 None —— 那会让 types.MethodType(func, None) 报
#           TypeError: instance must not be None
#   FIX-3 兜底 + 写后回读校验：其它未知不可序列化叶子丢成 None 并打印；
#         save() 写完立刻回读一次，避免"能存不能读"的静默损坏。
#
# 幂等，可反复运行；原文件备份为 .orig。QFLX_PICKLE_STRICT=1 关闭兜底（严格模式）。
# 同源的独立脚本（同目录）：setup/patch_persistence.py
API_PATCH="未执行"
PERS="$QROOT/relax/utils/persistence.py"
if [ -x "$VENV/bin/python" ] && [ -f "$PERS" ]; then
  PATCHER="$QROOT/.qflex_patch_persistence.py"
  cat > "$PATCHER" <<'QFLX_PERSISTENCE_PATCH_EOF'
#!/usr/bin/env python3
"""幂等修补 Qflex 的 relax/utils/persistence.py，使其兼容 jax 0.4.35+ / 0.4.38。

三处修补（均在本机 jax 0.4.38 上实测定界并验证）：

  FIX-1 named_shape
      0.4.35 起 ShapeDtypeStruct.named_shape 属性被移除（0.4.27 尚存）
      → 改用 getattr(x, "named_shape", {})，语义等价。

  FIX-2 jax 配置 flag（True 真凶）
      jax._src.config.State 实例（79 个配置 flag）基类是 pybind11 的
      xla_extension.config.Config，CPython 无法 pickle：
          TypeError: cannot pickle 'State' object
      它在真实 jaxpr 图里以「绑定方法」形式出现（State.__call__，经
      jax._src.compute_on.extend_compute_type 挂进 JaxprEqnContext），所以
      **不能丢成 None** —— 否则反序列化时 types.MethodType(func, None) 会报
          TypeError: instance must not be None
      正确做法：按名字持久化（"S:<属性名>"），反序列化时取回当前进程里的同名活
      对象；与文件里已有的 jax.core.Primitive 按名恢复（"P:<name>"）同一思路。

  FIX-3 兜底与回读校验
      ① Pickler.save 捕获其它 "cannot pickle ..." 叶子并打印（丢成 None），
         使训练不再在 trainer.setup() 阶段中断；QFLX_PICKLE_STRICT=1 可关闭。
      ② PersistFunction.save 写完后立即回读校验，避免"能存不能读"的静默损坏。

用法： python3 patch_persistence.py <path/to/persistence.py> [--check]
"""
import argparse
import py_compile
import sys
from pathlib import Path

MARK_F1 = '_ns = lambda x: getattr(x, "named_shape", {})'
MARK_F2 = "_JAX_CONFIG_STATE_MODULE"
MARK_F3 = "write_ok = False"

FIX1_OLD = """        assert all(x.named_shape == {} and x.sharding is None for x in in_descr_flat)
        assert all(x.named_shape == {} and x.sharding is None for x in out_descr_flat)"""

FIX1_NEW = """        # FIX-1: jax>=0.4.35 移除了 ShapeDtypeStruct.named_shape（0.4.27 尚存）
        _ns = lambda x: getattr(x, "named_shape", {})  # noqa: E731
        assert all(_ns(x) == {} and x.sharding is None for x in in_descr_flat)
        assert all(_ns(x) == {} and x.sharding is None for x in out_descr_flat)"""

# ---- FIX-3 的 save 回读校验：替换 PersistFunction.save 方法体 ----
SAVE_OLD = """    def save(self, path: os.PathLike):
        pickler = PersistFunctionPickler(open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(self)"""

SAVE_NEW = """    def save(self, path: os.PathLike):
        with open(path, "wb") as f:
            pickler = PersistFunctionPickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)
        # FIX-3②: 回读校验 —— 这一份结构稍后要由评估子进程用 Unpickler 读回，
        # 因此"能存"不等于"能用"。这里立刻读一次，失败就当场报明确错误。
        write_ok = False
        try:
            PersistFunction.load(path)
            write_ok = True
        except Exception as exc:  # pragma: no cover
            if os.environ.get("QFLX_PICKLE_STRICT") == "1":
                raise
            print(
                f"[persistence] WARNING 写出的结构无法回读（评估子进程会失败）: {path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return write_ok"""

FIX2_OLD = """class PersistFunctionPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, jax.core.Primitive):
            if obj.name in BLACKLIST:
                raise pickle.PickleError(f"Cannot pickle {obj.name}")
            return f"P:{obj.name}"
        elif isinstance(obj, jaxlib.xla_client.Traceback):
            return "T"
        elif isinstance(obj, types.FunctionType) and "<locals>" in obj.__qualname__:
            return "C"
        return None"""

FIX2_BLOCK = '''# FIX-2: jax 0.4.38 的配置 flag（jax._src.config.State 实例，基类是 pybind11 的
# xla_extension.config.Config，tp_new 缺失）无法 pickle：
#     TypeError: cannot pickle 'State' object
# 它在被保存的 jaxpr 里以「绑定方法」出现（State.__call__ <- JaxprEqnContext /
# jax._src.compute_on.extend_compute_type）。因此不能丢成 None —— 反序列化时
# types.MethodType(func, None) 会抛 "instance must not be None"。
# 这里采用与 jax.core.Primitive 完全一致的做法：按名字持久化，读回时取当前进程
# 的同名活对象（配置 flag 本就是进程级单例）。
_JAX_CONFIG_STATE_TYPES: tuple = ()
_JAX_CONFIG_STATE_MODULE = None
try:  # pragma: no cover
    from jax._src import config as _jax_config_state_mod  # type: ignore
    if hasattr(_jax_config_state_mod, "State"):
        _JAX_CONFIG_STATE_TYPES = (_jax_config_state_mod.State,)
        _JAX_CONFIG_STATE_MODULE = _jax_config_state_mod
except Exception:
    pass


def _jax_config_state_key(obj) -> str:
    """给出 flag 对象在 jax._src.config 里的属性名（用于按名持久化）。"""
    if _JAX_CONFIG_STATE_MODULE is not None:
        for k, v in list(vars(_JAX_CONFIG_STATE_MODULE).items()):
            if v is obj:
                return k
    for attr in ("__name__", "_name"):
        name = getattr(obj, attr, None)
        if isinstance(name, str) and name:
            return name
    return ""


class PersistFunctionPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, jax.core.Primitive):
            if obj.name in BLACKLIST:
                raise pickle.PickleError(f"Cannot pickle {obj.name}")
            return f"P:{obj.name}"
        elif isinstance(obj, jaxlib.xla_client.Traceback):
            return "T"
        elif _JAX_CONFIG_STATE_TYPES and isinstance(obj, _JAX_CONFIG_STATE_TYPES):
            key = _jax_config_state_key(obj)
            if key:
                return f"S:{key}"
            return "C"
        elif isinstance(obj, types.FunctionType) and "<locals>" in obj.__qualname__:
            return "C"
        return None

    # FIX-3①: 兜底 —— 其它未知的 "cannot pickle ..." 叶子丢成 None 并打印，
    # 避免整次训练在 setup 阶段被打断。QFLX_PICKLE_STRICT=1 可关掉。
    def save(self, obj, save_persistent_id=True):
        try:
            return super().save(obj, save_persistent_id)
        except TypeError as exc:
            if not str(exc).startswith("cannot pickle"):
                raise
            if os.environ.get("QFLX_PICKLE_STRICT") == "1":
                raise
            t = type(obj)
            print(
                "[persistence] dropped unpicklable object "
                f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', t)}: {exc}",
                flush=True,
            )
            return super().save(None, save_persistent_id)'''

UNPICK_OLD = """    def persistent_load(self, pid):
        if not isinstance(pid, str):
            raise pickle.UnpicklingError(f"Invalid persistent id {pid}")
        if pid == "T" or pid == "C":
            return None
        elif pid.startswith("P:"):"""

UNPICK_NEW = """    def persistent_load(self, pid):
        if not isinstance(pid, str):
            raise pickle.UnpicklingError(f"Invalid persistent id {pid}")
        if pid == "T" or pid == "C":
            return None
        elif pid.startswith("S:"):
            # FIX-2: 取回当前进程里同名的 jax 配置 flag（进程级单例）
            key = pid[2:]
            if _JAX_CONFIG_STATE_MODULE is not None:
                flag = getattr(_JAX_CONFIG_STATE_MODULE, key, None)
                if flag is not None:
                    return flag
            raise pickle.UnpicklingError(f"Unknown jax config flag: {key}")
        elif pid.startswith("P:"):"""


def patch(path: Path, check_only: bool = False) -> int:
    src = path.read_text(encoding="utf-8")
    orig = src
    notes = []

    def sub(old, new, mark, name):
        nonlocal src
        if mark in src:
            notes.append(f"{name} 已存在")
        elif old not in src:
            notes.append(f"{name} 锚点未找到!!")
        else:
            src = src.replace(old, new, 1)
            notes.append(f"{name} 已应用")

    if check_only:
        for mk, nm in ((MARK_F1, "FIX-1"), (MARK_F2, "FIX-2"), (MARK_F3, "FIX-3")):
            print(f"  {nm}: {'已应用' if mk in src else '缺失'}")
        return 0

    sub(FIX1_OLD, FIX1_NEW, MARK_F1, "FIX-1 named_shape")
    sub(FIX2_OLD, FIX2_BLOCK, MARK_F2, "FIX-2 config State 按名持久化 + FIX-3① 兜底")
    sub(UNPICK_OLD, UNPICK_NEW, "Unknown jax config flag", "FIX-2 unpickler")
    sub(SAVE_OLD, SAVE_NEW, MARK_F3, "FIX-3② 写后回读校验")

    if src != orig:
        backup = Path(str(path) + ".orig")
        if not backup.exists():
            backup.write_text(orig, encoding="utf-8")
            print(f"[OK] 备份原文件 -> {backup}")
        path.write_text(src, encoding="utf-8")

    # 用 compile() 而不写 .pyc，避免在被修补目录里留下 __pycache__
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    final = path.read_text(encoding="utf-8")
    print(f"[OK] 语法校验通过: {path}")
    for n in notes:
        print("   -", n)
    for mk, nm in ((MARK_F1, "FIX-1"), (MARK_F2, "FIX-2"), (MARK_F3, "FIX-3")):
        print(f"  {nm}: {'已应用' if mk in final else '缺失'}")
    missing = [n for n in notes if "未找到" in n]
    return 1 if missing else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = Path(a.path)
    if not p.exists():
        print(f"[FAIL] 文件不存在: {p}")
        sys.exit(2)
    sys.exit(patch(p, a.check))
QFLX_PERSISTENCE_PATCH_EOF
  if "$VENV/bin/python" "$PATCHER" "$PERS"; then
    API_PATCH="已应用（FIX-1 named_shape / FIX-2 config State / FIX-3 兜底+回读校验）"
  else
    API_PATCH="★失败（源码锚点与预期不符，需人工处理）"
    no "persistence.py 补丁失败 —— 请把 $PERS 发我"
  fi
  rm -f "$PATCHER"
elif [ ! -f "$PERS" ]; then
  API_PATCH="★跳过（找不到 $PERS）"
else
  API_PATCH="★跳过（$VENV/bin/python 不可用）"
fi

# ---------------------------------------------------------------------
if [ "$MODE" = "all" ] || [ "$MODE" = "deps" ]; then
  if [ "$GPU_OK" = 1 ] || [ "$MODE" = "deps" ]; then
    say "4) 安装其余依赖（含编译 relax 的 C 扩展: futex / prctl / spinlock）"

    # ★ 关键：把选定的 jax spec 一起交给 resolver。
    # 实测 optax 最新版要 jax>=0.5.3、numpyro 最新版要 jax>=0.7.0，
    # 而 base.txt 两个都没锁 —— 不钉住 jax 的话，装了 0.4.38 会立刻被顶掉。
    JAX_PIN=()
    [ -n "$JAX_USED" ] && JAX_PIN=("$JAX_USED")

    DEPS_OK=0
    echo "-- 尝试 A: 钉住 jax($JAX_USED) 安装依赖（让 uv 回溯 numpyro/optax 旧版）"
    if [ -n "${CFLAGS:-}" ]; then
      info "已从环境继承 CFLAGS，尝试 A 直接带上：$CFLAGS"
    else
      info "注：Ubuntu 26.04 的 GCC 14+ 会在 src/futex.c 报 -Wincompatible-pointer-types"
      info "    （旧 GCC 只是 warning，源码逻辑本身正确）→ 失败后自动走回退 B，"
      info "    那一大段红色报错可以忽略。想一次过就：export CFLAGS=\"$GCC_RELAX_DOC\""
    fi
    if uv pip install --python "$VENV" "${JAX_PIN[@]}" -r requirements/uv/relax_myosuite.txt; then
      ok "依赖安装完成（jax 保持钉版）"
      DEPS_OK=1
    else
      # 回退 B/C: 放宽新 GCC 的诊断等级。
      # Ubuntu 26.04 的 GCC（GCC 14+）把 -Wincompatible-pointer-types 从
      # warning 提升为 error，而 src/futex.c 恰好有 4 处签名/实参类型不匹配
      # （函数声明收 struct lock_t*，实际传 atomic_uint*；逻辑本身是对的，
      #  函数体立刻用 LOCK_COUNT(lock) 按结构体布局取回字段）。
      # ★ 注意 CFLAGS='-std=gnu11' 是**无效**的（这是类型诊断，不是 C 标准
      #   版本问题）—— v4 的这条回退已被实测否掉。
      warn "钉版安装失败（多半是 C 扩展编译），试放宽 GCC 诊断等级"
      GCC_RELAX="$GCC_RELAX_DOC"
      info "回退 B: CFLAGS='$GCC_RELAX'"
      if CFLAGS="$GCC_RELAX" uv pip install --python "$VENV" "${JAX_PIN[@]}" -r requirements/uv/relax_myosuite.txt; then
        ok "依赖安装完成（C 扩展按放宽诊断编译）"
        DEPS_OK=1
        CFLAGS_USED="B: $GCC_RELAX"
      else
        info "回退 C: CFLAGS='-O2 -fPIC -w'（全警告抑制）"
        if CFLAGS="-O2 -fPIC -w" uv pip install --python "$VENV" "${JAX_PIN[@]}" -r requirements/uv/relax_myosuite.txt; then
          ok "依赖安装完成（C 扩展以 -w 编译）"
          DEPS_OK=1
          CFLAGS_USED="C: -O2 -fPIC -w"
        fi
      fi
    fi

    if [ "$DEPS_OK" != 1 ]; then
      no "依赖安装失败"
      warn "手动兜底（按顺序试）:"
      info "1) CFLAGS='-O2 -fPIC -DNDEBUG -std=gnu11 -Wno-error -Wno-incompatible-pointer-types' \\"
      info "     uv pip install --python $VENV '$JAX_USED' -r requirements/uv/relax_myosuite.txt"
      info "   （Ubuntu 26.04 的 GCC(GCC 14+) 把 src/futex.c 的指针类型不匹配当 error，"
      info "     上面的 flag 把它降回 warning；CFLAGS='-std=gnu11' 已实测无效）"
      info "2) 若报 numpyro/optax 与 jax 冲突，显式钉住实测兼容组合:"
      info "   uv pip install --python $VENV '$JAX_USED' 'numpyro==0.19.0' 'optax==0.2.5' -r requirements/uv/relax_myosuite.txt"
      info "   （0.19.0 / 0.2.5 是实测与 jax 0.4.38 兼容的上界；压旧无害，"
      info "     Qflex 只用 numpyro.distributions.Normal 和 optax 的标准优化器）"
      info "3) ★不要★为了装依赖放开 jax —— 实测不钉 jax 时会被顶到 0.7.0+"
    fi

    # -----------------------------------------------------------------
    # 4.5 复验：第 4 步是否偷偷换掉了 jax
    # ----------------------------------------------------------------
    say "4.5) 复验 jax 是否被依赖安装改动"
    write_probe
    if [ -x "$VENV/bin/python" ]; then
      POST_OUT="$(run_probe)"
      POST_VER="$(probe_ver "$POST_OUT")"
      printf '%s\n' "$POST_OUT" | sed 's/^/      /'
      echo ""
      info "第 3 步选定 : ${JAX_USED_VER:-（未知）}"
      info "第 4 步之后 : ${POST_VER:-（读不到）}"

      if [ "$POST_VER" = "$JAX_USED_VER" ] && probe_has_gpu "$POST_OUT"; then
        JAX_FINAL_OK=1
        ok "jax 未被改动，GPU 仍可用 —— 环境一致"
      else
        no "jax 状态被改变了（版本不符或 GPU 丢失）→ 自动回滚"
        if [ -n "$JAX_USED" ]; then
          info "回滚命令: uv pip install --python $VENV '$JAX_USED'"
          uv pip install --python "$VENV" "$JAX_USED"
          RB_OUT="$(run_probe)"
          RB_VER="$(probe_ver "$RB_OUT")"
          printf '%s\n' "$RB_OUT" | sed 's/^/      /'
          if [ "$RB_VER" = "$JAX_USED_VER" ] && probe_has_gpu "$RB_OUT"; then
            JAX_FINAL_OK=1
            JAX_ROLLBACK="done"
            ok "回滚成功，jax 回到 $RB_VER 且 GPU 可用"
            warn "注意: 回滚后 numpyro/optax 可能又与新 jax 冲突，第 5 步冒烟会暴露"
          else
            no "回滚也没修好 —— 把上面输出发我，别继续跑训练"
          fi
        else
          no "没有可用的 jax spec 记录，无法自动回滚"
        fi
      fi
    else
      no "venv 不存在，跳过复验"
    fi
  else
    warn "GPU 未就绪，跳过依赖安装（GPU 好了以后跑: bash ~/setup_qflex.sh deps）"
  fi
fi

# ---------------------------------------------------------------------
if [ "$MODE" = "all" ] || [ "$MODE" = "smoke" ]; then
  if [ -x "$VENV/bin/python" ]; then
    say "5) 冒烟测试: C 扩展 + 完整 import 链 + 向量环境 + myosuite + JAX"
    # 注意：不设 MUJOCO_GL —— 建环境/走步不需要 GL，设错反而会误报失败
    ( cd "$QROOT" && "$VENV/bin/python" - <<'PYEOF'
def sec(t): print("\n--- %s ---" % t)
import traceback

sec("C 扩展（单独 import）")
for m in ("relax.futex", "relax.prctl", "relax.spinlock"):
    try:
        __import__(m); print("  OK  ", m)
    except Exception:
        print("  FAIL", m); traceback.print_exc()

sec("完整 import 链: relax.env / relax.env.vector / process3 / envs")
# relax/env/__init__.py:20 `import envs`
# relax/env/__init__.py:22 -> relax.env.vector -> vector/__init__.py:6
#   -> process3.py:14 `import envs`, :17 `from relax.futex import ...`
try:
    import relax.env
    from relax.env import create_vector_env
    import relax.env.vector.process3
    print("  OK   relax.env / create_vector_env / process3 全部导入成功")
except Exception:
    print("  FAIL 完整 import 链"); traceback.print_exc()

sec("真起 futex 向量环境（2 进程，走 3 步）—— 覆盖 process3 + futex + prctl 全链路")
try:
    import numpy as np
    venv_, obs_dim, act_dim = create_vector_env(
        "myoLegWalk-v0", 2, seed=0, action_seed=0, mode="futex")
    print("  obs_dim:", obs_dim, "| act_dim:", act_dim)
    out = venv_.reset()
    obs = out[0] if isinstance(out, tuple) else out
    print("  reset obs:", getattr(obs, "shape", type(obs)))
    for i in range(3):
        obs, rew, te, tr, inf = venv_.step(
            np.zeros((2, act_dim), dtype=np.float32))
    print("  3 步 OK | obs:", np.asarray(obs).shape,
          "| rew:", np.asarray(rew).shape, "| term:", np.asarray(te).shape)
    venv_.close()
    print("  close OK")
except Exception:
    print("  FAIL futex 向量环境"); traceback.print_exc()

sec("myosuite 单环境")
try:
    import myosuite
    print("  myosuite:", getattr(myosuite, "__version__", "?"))
    import gymnasium as gym
    env = gym.make("myoLegWalk-v0")
    obs, info = env.reset(seed=0)
    print("  obs:", getattr(obs, "shape", type(obs)), "| act:", env.action_space.shape)
    r = None
    for _ in range(5):
        obs, r, te, tr, info = env.step(env.action_space.sample())
    print("  5 步随机动作 OK, reward =", float(r))
    env.close()
except Exception:
    print("  FAIL myosuite"); traceback.print_exc()

sec("JAX")
try:
    import jax, numpy as np
    print(" ", jax.__version__, "| numpy", np.__version__,
          "|", [str(d) for d in jax.devices()])
except Exception:
    traceback.print_exc()
PYEOF
    )

    # 渲染单独开一个进程测（mu 的 GL 后端在 import 时就定死了，必须新进程 + 环境变量）
    echo ""
    echo "--- 离屏渲染 EGL（仅信息，训练不需要） ---"
    MUJOCO_GL=egl "$VENV/bin/python" - <<'PYEOF'
try:
    import mujoco
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 64, 64)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    print("  EGL 渲染 OK, frame:", r.render().shape)
except Exception as e:
    print("  EGL 渲染不可用:", type(e).__name__, str(e)[:200])
    print("  （只影响 --record_video，不影响训练）")
PYEOF
  else
    no "venv 不存在，跳过冒烟"
  fi
fi

# ---------------------------------------------------------------------
say "汇总"
cat > /tmp/_qflex_versions.py <<'PYEOF'
import importlib.metadata as m
for p in ("jax", "jaxlib", "numpy", "numpyro", "optax", "dm-haiku",
          "myosuite", "mujoco", "gymnasium", "numba"):
    try:
        print("%-10s %s" % (p, m.version(p)))
    except Exception:
        print("%-10s (未安装)" % p)
PYEOF
echo "  uv              : $(uv --version 2>/dev/null)"
echo "  venv python     : $([ -x "$VENV/bin/python" ] && "$VENV/bin/python" -V 2>&1)"
echo "  选定索引        : ${UV_DEFAULT_INDEX:-未知}"
echo "  GPU_OK          : $GPU_OK"
echo "  jax 探测方式    : $([ "$JAX_FAST" = 1 ] && echo '快速通道（沿用 .jax_version_ok，未重装）' || echo '完整候选循环')"
REJ_LIST=""
if [ -f "$JAX_REJECTED" ]; then
  while IFS= read -r L; do [ -n "$L" ] && REJ_LIST="$REJ_LIST$L "; done < "$JAX_REJECTED"
fi
echo "  实测失败已跳过  : ${REJ_LIST:-（无）}"
echo "  第3步选定 jax   : ${JAX_USED:-（未确定）}$([ "$JAX_UNPINNED" = 1 ] && echo '  ★无版本约束，兼容性存疑')"
echo "  依赖后 jax 一致 : $([ "$JAX_FINAL_OK" = 1 ] && echo '是' || echo '否/未测') ${JAX_ROLLBACK:+（已自动回滚）}"
echo "  C扩展编译参数   : ${CFLAGS_USED:-默认（一次过，未走回退）}"
echo "  API兼容补丁     : ${API_PATCH:-未检查}"
echo "  --- 已装版本 ---"
if [ -x "$VENV/bin/python" ]; then
  "$VENV/bin/python" /tmp/_qflex_versions.py 2>&1 | sed 's/^/  /'
else
  echo "  （venv 不存在）"
fi
echo "  venv 路径       : $VENV"
echo "  日志            : ~/setup_qflex.log"
echo ""
if [ "$GPU_OK" = 1 ] && { [ "$JAX_FINAL_OK" = 1 ] || [ "$MODE" = "jax" ]; }; then
  ok "环境就绪。下一步跑训练冒烟（会同时给出实测吞吐）:"
  echo "     cd $QROOT"
  echo "     export XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false"
  echo "     time $VENV/bin/python scripts/train.py --alg qflex --env myoLegWalk-v0 --seed 100 --total_step 100000 --num_vec_envs 80 --suffix smoke1e5"
  echo ""
  info "关于这次冒烟的时长与产物（已逐行核对源码，非估计）:"
  info "  * 训练目录: $QROOT/logs/myoLegWalk-v0/qflex_<时间>_s100_smoke1e5/"
  info "  * 训练侧每 save_policy_every = min(1e6, total_step/20) 步存一个 policy-*.pkl"
  info "    并通知 evaluator（off_policy.py:302/:87）→ **评估次数恒为 20 次**，"
  info "    所以 total_step 越小，评估在总时长里占比越大。"
  info "  * evaluator 是独立 CPU 子进程（evaluator.py:8 显式设了 JAX_PLATFORMS=cpu，"
  info "    **不占显存**），每次评估 5 局，结果写 <训练目录>/log.csv"
  info "    （列: step,avg_ret,std_ret）—— 这就是学习曲线的原始数据。"
  info "  * 首次 update 会做一次大 JIT（lax.scan 20 步 + vmap/grad），可能几分钟"
  info "    没有新输出，属正常，别以为卡死。"
  info "  * 想看显存请另开一个终端: watch -n 2 nvidia-smi（WSL 里 watch 可用）"
else
  no "环境未就绪，见上面 [FAIL] 行"
fi
