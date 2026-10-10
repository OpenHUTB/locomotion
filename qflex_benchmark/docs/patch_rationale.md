# 复现说明：我们的命令与官方口径的差异审计

> 用途：交付报告的「复现说明 / 局限（Limitations）」一节。
> 审计时间：2026-09-26 22:00｜审计对象：`scripts/train.py` + 4 格正式批量命令
> 方法：从 GitHub 上游拉原始文件做逐行基线（本地仓库不是 git checkout，无 `git status`）

## 1. 代码层面：本地 `train.py` = 上游 + 3 处纯增量补丁

| 文件 | 行数 | md5 | 说明 |
|---|---|---|---|
| 上游 `LNSGroup/Qflex@main:scripts/train.py` | 319 | — | 从 `raw.githubusercontent.com` 直接取证 |
| 本地 `scripts/train.py.orig` | 319 | `409a0faec6c44906b585d3e8bcee7877` | **未改动的留档** |
| 本地 `scripts/train.py`（在用） | 329 | `2a6e810db0490656a9102b9507b0e81b` | = 上游 + 下述 3 处 |

三处补丁（`diff train.py.orig train.py` 全文只有这些）：

| # | 上游原文 | 本地 | 作用 |
|---|---|---|---|
| 1 | —（无此参数） | `:92-97` 新增 `--save_policy_every`（默认 0）与 `--num_workers`（默认 0） | 暴露两个此前写死的工程开关 |
| 2 | `create_vector_env(..., mode="futex")` | `:114-117` 改为 `vec_kwargs = {"num_workers": ...} if >0 else {}` + `**vec_kwargs` | 只在显式传值时才改变行为 |
| 3 | `save_policy_every=min(int(1e6), int(args.total_step / 20))` | `:310-312` 改为可被 CLI 覆盖，否则回落到原公式 | 同上 |

**结论：补丁是纯增量、默认值保持原版行为；未删除或改写任何上游逻辑。**

## 2. 参数层面：显式传的值与上游默认值数值等价

| 参数 | 上游行为 | 我们传的值 | 是否等价 |
|---|---|---|---|
| `save_policy_every` | 写死 `min(1e6, total_step/20)` | `5e6 → 250000`；`2e7 → 1000000` | ✅ 与公式逐值相等 |
| `num_vec_envs` | README 指定 70 / 80 | 70 / 80 | ✅ |
| `total_step` | README 指定 | 5e6 / 5e6 / 2e7 / 5e6 | ✅ |
| `seed` | README 指定 100 | 100 | ✅ |
| 其余超参 | argparse 默认值 | 不传 | ✅ |
| **`--num_workers`** | **上游无此参数** ⇒ `process3.py:23-24` 令 `num_workers = num_envs` | 显式给 14 / 20 | ⚠️ **刻意偏离，见 §4** |
| **`--record_video`** | README 示例命令带；Notes 明说可选 | 不带 | ⚠️ **刻意偏离，见 §4** |

## 3. README 官方命令 vs 我们的命令（4 格逐条对照）

| env | README（`README.md:170-184`） | 我们 | 差异 |
|---|---|---|---|
| `h1-run-v0` | 5e6 / 70 env / `--record_video` | 5e6 / 70 / w=14 / 无 video | +workers, −video |
| `h1-balance_simple-v0` | 2e7 / 70 env / `--record_video` | 2e7 / 70 / w=14 / 无 video | +workers, −video |
| `SMPLHumanoidJump-v1` | 5e6 / 80 env / `--record_video` | 5e6 / 80 / w=20 / 无 video | +workers, −video |
| `myoHandPenTwirlRandom-v0` | 5e6 / 80 env / `--record_video` | 5e6 / 80 / w=20 / 无 video | +workers, −video |

`--save_policy_every` 不进差异列（与官方默认等价，见 §2）。

## 4. 两处刻意偏离的理由（须写进 Limitations）

### (a) 增加 `--num_workers`（必需，非可选）
上游不传该参数时 `relax/env/vector/process3.py:23-24` 会令 `num_workers = num_envs`
⇒ 官方 h1 命令会起 **70 个子进程各带 1 个 env**，内存不足的机器上直接 OOM（70 × ~440 MB ≈ 30 GB；
本机初始 WSL 只分到 ~15.9 GB 时即被 OOM killer 杀死，须把 WSL2 上限提到 28 GB）。

**它不改变学习语义的源码依据**：`process3.py:61-65` 由**父进程**按 `num_envs` 一次性生成 seeds 数组；
`:87-94` 每个 worker 只接收自己那段**下标**与**对应下标**的 seed。
⇒ env 下标 ↔ seed 的映射与 worker 数无关，worker 数只影响"哪些 env 在同一个进程里被 step"。
（未做逐位轨迹对比实验，属源码级论证。）

### (b) 不加 `--record_video`
README「Notes」原文：*"Add `--record_video` only when you want evaluator videos; otherwise the
non-video evaluator is used."* ⇒ 官方明确它是可选项，不影响训练，只影响评估阶段是否录视频。
取"不加"是为了与**已完成并可对照的 MyoLeg-Walk 那一格**保持完全一致（该格同样未加）。

## 5. 由本次审计顺带确认的两条硬事实（对可行性判断有决定性影响）

1. **`size=int(1e6)`（`train.py:126`）是上游硬编码**，26 个 CLI 参数中无任何 buffer 容量开关。
   ⇒ 大维度环境（Ostrich `|S|=596, |A|=120` ⇒ 满仓 5.25 GB；MS700 `3148/700` ⇒ 28.0 GB）
   的 replay 内存**无法靠调参绕过**，只能改这一行代码。
2. `relax/algorithm/` 实有 **14** 个算法模块，但 README 只把 **6** 个（qflex + sdac/dacer/qsm/crossq/sac）
   列为可用 baseline；论文 9 算法里的 DynSyn / Lattice / DEP-RL 在本仓库**无实现**。
