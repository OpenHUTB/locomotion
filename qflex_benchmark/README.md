# Qflex Benchmark

使用 Qflex 官方代码运行基准任务并生成学习曲线。入口脚本会自动克隆上游仓库、应用所需补丁、配置环境、运行训练并出图。

## 快速开始

在 WSL2 Ubuntu 中进入本目录：

```bash
bash main.sh check      # 检查系统依赖、GPU、内存和磁盘
bash main.sh setup      # 克隆 Qflex、应用补丁并安装环境
bash main.sh verify     # 验证环境
bash main.sh train      # 开始训练
bash main.sh plot       # 根据训练日志出图
```

也可以显式运行完整流程：`bash main.sh all`。完整默认训练约需 300 小时；脚本会在开始前显示计划并等待确认。无参数运行 `bash main.sh` 只显示帮助，不会启动训练。

常用选项：

```bash
bash main.sh train --seed 100       # 只跑一个种子
bash main.sh train --no-ostrich      # 跳过 Ostrich，运行其余五个任务
bash main.sh all --dry-run           # 打印计划，不执行流程
```

长任务会在 tmux 中运行，日志保存在 `~/qflex_logs/`。训练中断后重跑相同命令，已完成的 run 会跳过。运行 `bash main.sh --help` 查看全部选项。

## 环境要求

- WSL2 Ubuntu、Git、Python、tmux、可用的 NVIDIA GPU 和联网环境。
- 默认包含 Ostrich 任务，需要 WSL 可用内存约 28 GB；可在 Windows 用户目录的 `.wslconfig` 中设置 `memory=28GB` 和 `swap=16GB`，然后执行 `wsl --shutdown`。
- 内存不足时用 `--no-ostrich` 跳过 Ostrich。训练时间和资源占用取决于硬件。

## 结果与范围

本流水线覆盖仓库中已实现的 6 个任务、6 种算法和种子 100、101、102，共 108 个 run。已在 WSL2 Ubuntu、RTX 5070 Ti Laptop（12 GB）和 32 GB 物理内存环境完成这一配置。

结果保存在 Qflex 的 `logs/` 下；图和汇总 CSV 默认输出到本目录的 `figs/`。论文只提供曲线图片，数值比较有读图误差；本结果使用 3 个种子，不能直接与论文的 5 次运行误差带比较。MS-Human-700 各任务的性能评估仍需单独运行和记录。

## 实现说明

- `main.sh` 是统一入口；`setup/`、`run/` 和 `plot/` 保存各阶段脚本。
- `patches/` 中的补丁由入口自动应用到克隆的 Qflex，不需要手动执行 `git apply`。
- 训练参数和算法调用沿用现有配置；本目录的补丁用于暴露运行参数及兼容较新 JAX。
- 补丁原因与已知问题见 `docs/patch_rationale.md` 和 `docs/known_issues.md`。
