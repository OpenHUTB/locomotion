# patches/

两个补丁都是**对 Qflex 官方仓库**的改动，请在 Qflex 仓库根目录 `git apply`：

```bash
git clone --recurse-submodules https://github.com/LNSGroup/Qflex.git ~/work/Qflex
cd ~/work/Qflex
git apply <path>/0001-train-expose-num-workers-and-save-policy-every.patch
git apply <path>/0002-persistence-jax0438-compat.patch
```

| 补丁 | 目标文件 | 作用 | 是否必需 |
|---|---|---|---|
| `0001-train-expose-num-workers-and-save-policy-every.patch` | `scripts/train.py` | 暴露 `--num_workers`（默认 0 = 原行为）与 `--save_policy_every`（默认 0 = 原公式） | **必需**：不暴露 `--num_workers`，h1 两格在内存不足的机器上必 OOM（70 worker × ~440 MB ≈ 30 GB） |
| `0002-persistence-jax0438-compat.patch` | `relax/utils/persistence.py` | 兼容 `jax>=0.4.35` | **在 sm_120 等新卡上必需**：新卡必须用 `jax>=0.4.38` |

**性质**：两者都是**纯增量补丁**，默认值完全保持官方行为，未删除或改写任何官方逻辑。
`0001` 的另一重作用是把官方写死的两个"工程开关"变成可显式传参，便于把偏离项写在命令行里接受审查。

- 逐行理由、与官方 README 命令的差异审计：`../docs/patch_rationale.md`
- 补丁的上游侧取自 `LNSGroup/Qflex@main` 的 `scripts/train.py`（md5 `409a0faec6c44906b585d3e8bcee7877`）
  与 `relax/utils/persistence.py`；两个补丁均已用 `git apply --check` 在真实上游仓库上验证可干净套用。
- `0002` 的脚本化等价物 = `../setup/patch_persistence.py`（幂等，更适合已装好的环境）。
