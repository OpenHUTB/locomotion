# 实测踩到的坑与处置

全部来自本机（WSL2 Ubuntu / RTX 5070 Ti Laptop sm_120 / 物理 32 GB，WSL2 上限 28 GB）的实跑日志。

## 1. 新卡（sm_120 / Blackwell）上 GPU 起不来

**现象**：`XlaRuntimeError: INTERNAL: ptxas exited with non-zero error code 65280,
ptxas fatal : Program with .target 'sm_90a' cannot be compiled to future architecture`

**根因**：官方 `requirements/uv/jax-cuda12.txt` 锁 `jax[cuda12]==0.4.27`，该版本自带的 `ptxas`
只认到 `sm_90a`，无法为更新的架构编译。

**处置**：必须 `jax>=0.4.38`；随之需要 `patches/0002`（见第 2 条）。
`setup/setup_qflex.sh` 内置版本探测：逐个候选实测一次矩阵乘，通过的记入 `.jax_version_ok`，
失败的记入 `.jax_rejected`（重跑自动跳过，`FORCE_JAX_PROBE=1` 可强制重试）。
**换到非 Blackwell 机器时删掉这两个文件**，0.4.27 会重新参与探测。

## 2. `jax>=0.4.35` 后 `persistence.py` 崩溃

**现象 A**：`AttributeError: 'ShapeDtypeStruct' object has no attribute 'named_shape'`（0.4.35 起该属性被移除）

**现象 B**：`TypeError: cannot pickle 'State' object`
（`jax._src.config.State` 基类是 pybind11 对象，CPython 无法 pickle；它在 jaxpr 图里以**绑定方法**出现，
不能简单丢成 `None` —— 那样反序列化会报 `TypeError: instance must not be None`）

**处置**：`patches/0002-persistence-jax0438-compat.patch`，三处修补：

1. `named_shape` 改 `getattr(x, "named_shape", {})`；
2. 该 `State` 对象**按名字持久化**（`"S:属性名"`），读回时取当前进程里的同名活对象
   （与文件里已有的 `jax.core.Primitive` 按名恢复同一思路）；
3. `PersistFunction.save` 写完立即回读校验，避免"能存不能读"的静默损坏（`QFLX_PICKLE_STRICT=1`
   可把警告升级为异常）。

等价的脚本化做法：`setup/patch_persistence.py <path/to/relax/utils/persistence.py>`（幂等）。

## 3. h1 组的 venv 依赖互斥

**现象**：`uv` 报 unsatisfiable，`relax_humanoidbench` 整套装不上。

**根因**：`envs/humanoid-bench/setup.py` 钉 `torch==2.3.1` ⇒ 要求 `nvidia-cudnn-cu12==8.9.2.26`；
而 `jax-cuda12-plugin 0.4.38` 要求 `nvidia-cudnn-cu12>=9.1,<10`，两者互斥。
这是"为了 Blackwell 把 jax 从 0.4.27 升到 0.4.38"引入的，不是原作者环境的问题。

**处置**：只给这个 venv 加 `--torch-backend=cpu`（`setup/setup_other_envs_v3.sh` 已内含；
单独补跑用 `setup/fix_humanoidbench.sh`）。依据：`envs/humanoid-bench/humanoid_bench/env.py`
全文不 import torch，H1 是 MuJoCo CPU 仿真，CPU 版 torch 够用。

## 4. 不传 `--num_workers` 必然 OOM

官方 `relax/env/vector/process3.py`：不传该参数时 `num_workers = num_envs`
⇒ 官方 h1 命令会起 70 个子进程、每个带 1 个 env（约 30 GB）。
本机 2026-09-24 02:25 被 OOM killer 实锤杀死（当时 WSL 只分到约 15.9 GB）；把 WSL2 上限提到 28 GB 后才跑通。
`patches/0001` 把它暴露成 CLI 参数。**这是本流水线唯一一处主动偏离官方命令的地方**，
源码论证见 `patch_rationale.md` §4(a)。

## 5. 内存是瓶颈，不是算力

实测峰值（正式跑批，非外推）：`SMPL 6,572 MB｜myoLeg 14,655｜h1-run 17,064｜h1-balance 17,264｜
myoHand 19,234｜Ostrich 24,708 MB`。

**内存是环境属性，跨环境外推不成立**：同为 80 env / w=20，SMPL 6.57 GB vs myoHand 19.23 GB，差 2.9 倍。
判"能不能两路并行"必须逐环境实测。

- Ostrich 必须 `WATCH_MB=1200`（默认 3000 会误杀），`run_envs_formal_v5.sh` 会在启用 Ostrich 时自动下调。
- Ostrich 默认关闭，需 `WITH_OSTRICH=1`。
- 断网不影响训练：`scripts/` 与 `relax/` 无网络调用，日志只写本地 `events.out.tfevents`。

## 6. 看门狗不识别时钟跳变（已知残留问题）

`run_envs_formal_v5.sh` 的单 run 硬超时（`RUN_TIMEOUT`，默认 36 h）按墙钟计。
中途停电或长时间睡眠后恢复，墙钟跳变量会超过阈值 ⇒ 当次 run 被判超时并整组 kill。

影响可控：已完成的 run 有 `.formal_done`，重跑同一条命令会自动跳过 ⇒ 不丢数据，只是被打断的那一个 run 要重跑。
更好的修法是让看门狗用单调时钟或识别跳变（尚未实现）。

## 7. `std_ret` 在三个环境里恒为 0

SMPL-Jump / MyoLeg-Walk / Ostrich-Run 的 `reset_noise_scale=0` ⇒ `env.reset()` 每次回同一初始姿态，
`get_deterministic_action` 又是确定性动作 ⇒ 5 局逐位相同，`log.csv` 的 `std_ret` 恒等于 0。

- 这不是"评估很稳"，而是**每个评估点只有 1 个有效样本**（该环境评估协议的属性）。
- 图上的误差带是**跨种子** std，与 `std_ret` 语义不同，不要混用。
- H1 两格与 MyoHand 的 `std_ret` 非零（它们的 reset 另带噪声）。

## 8. h1 两格只有 19 个评估点

`save_policy_every = min(1e6, total_step/20)`，而 h1 用 70 个并行环境，70 除不尽 10^k
⇒ 评估点自然落在 19 个上（其余 4 格是 20 个）。整除问题，不是失败；出图脚本把它写进脚注。

## 9. 图"看着比论文直"不是 bug

每条曲线只有 19~20 个评估点（官方口径），所以折线必然"直"；论文那条曲线的折点密度实测约为我们的 2~4 倍，
线宽 12~36 px（我们约 5 px）。这是出图口径差异，不影响任何训练数值。
只想让观感接近论文可用 `plot/export_6grid_paperlook.py`（加粗线 + y 轴不迁就误差带，数据未改）；
要让评估曲线本身变密只能重训并把 `save_policy_every` 调小，代价约等于整体重训，不建议。
