# 人的移动

[分层世界模型作为视觉全身类人生物控制器](https://www.nicklashansen.com/rlpuppeteer) 

<img src="assets/0.png" width="100%"></br>

[[网页]](https://www.nicklashansen.com/rlpuppeteer) [[论文]](https://arxiv.org/abs/2405.18418) [[模型]](https://drive.google.com/drive/folders/1cgt9HzquO5mcB71Krv0C0mD10scfMquO?usp=sharing)

----

## 概述

我们提出了木偶（Puppeteer，基于TD-MPC2:Temporal Difference Learning for Model Predictive Control），这是一种通过视觉观察的全身人类人体控制的分层世界模型。我们的方法会产生自然和类似人类的动作，而无需任何奖励设计或技能基元，并穿越了具有挑战性的地形。

<img src="assets/1.png" width="100%" style="max-width: 640px"><br/>

该存储库包含用于训练和评估低级（跟踪）和高级（操纵）世界模型的代码。我们开放两个层次结构的模型检查点，以便您可以在不训练任何模型的情况下入门。模型检查点可在 [此处](https://drive.google.com/drive/folders/1cgt9HzquO5mcB71Krv0C0mD10scfMquO?usp=sharing) 下载。

----

## 入门


使用项目的 [requirementx.txt](../requirements.txt) 安装软件包。

注意：

windows系统安装从 [链接](https://download.pytorch.org/whl/cu126/torch-2.8.0%2Bcu126-cp310-cp310-win_amd64.whl) 下载的pytorch的GPU版本

```shell
pip install torch-2.8.0%2Bcu126-cp310-cp310-win_amd64.whl
```

linux系统（ubuntu）安装从 [链接](https://download.pytorch.org/whl/cu126/torch-2.8.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=6bf3c0085af4176137f216c39995dede9beda9af1307fd1dee2305f4f351eb42)下载的pytorch的GPU版本

```shell
pip install torch-2.8.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl
```

您将需要一台带有GPU（> = 24 GB内存）的机器进行训练； CPU 和 RAM 使用微不足道。我们提供一个`Dockerfile`，以便于安装。您可以通过运行来构建Docker图像

```
cd docker && docker build . -t <user>/puppeteer:1.0.0
```

此 Docker 镜像包含运行训练和推理所需的所有依赖项。

----

## 支持的任务

该代码库当前支持**8**个使用 DMControl 在 Mujoco 实现的 CMU 类人体模型的全身控制任务。任务定义如下：

| 任务                    | 视觉
|-----------------------| --- |
| 站立 stand              | N
| 走路 walk               | N
| 跑步 run                | N
| 过道 corridor           | Y
| 跨栏过道 hurdles-corridor | Y
| 绕墙过道 walls-corridor        | Y
| 沟渠过道 gaps-corridor         | Y
| 楼梯过道 stairs-corridor  | Y

可以通过指定`train.py` and`evaluation.py`的`task`参数来运行。

## 示例用法

我们提供了有关如何评估我们提供的 Puppeteer 模型检查点的示例，以及如何在下面训练自己的木偶代理。

### 评估

请参阅下面有关如何评估下载的低层和高层检查点的示例。

```shell
python evaluate.py task=corridor low_level_fp=/path/to/tracking.pt checkpoint=/path/to/corridor-1.pt
# 增加了保存视频
# python evaluate.py task=corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/corridor-1.pt save_video=true
python evaluate.py task=gaps-corridor low_level_fp=/path/to/tracking.pt checkpoint=/path/to/gaps-1.pt
# python evaluate.py task=gaps-corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt checkpoint=D:/work/workspace/locomotion/human/model/gaps-1.pt save_video=true
# 下载的模型位于：D:/work/workspace/locomotion/human/model/gaps-corridor-1.pt
# Windows 平台下每秒刷新一次查看GPU使用率
nvidia-smi -l 1
```

所有高级检查点都经过相同的低级检查点训练。有关参数的完整列表，请参见`config.yaml`。


### 训练

请参阅下面的示例，介绍了如何训练木偶的低级和高级世界模型。我们建议在`config.yaml`中配置[权重和偏置](https://wandb.ai) (`wandb`) 以跟踪训练进度 （训练代码可以网页上查看，评估代码没有。测试代码请参考 [wandb_demo.py](./wandb_demo.py)）。


配置数据集

```
# 训练跟踪模型
python train.py task=tracking
python train.py task=walk low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt
python train.py task=corridor low_level_fp=D:/work/workspace/locomotion/human/model/tracking.pt
```

我们推荐所有任务使用默认的超参数。参数的完整链表请查看`config.yaml`。



----


## myochallenge_2025eval

[submit tutorials](https://github.com/MyoHub/myochallenge_2025eval/blob/main/tutorials/GHaction_Submission.md)

## 教AI来跑（NIPS 2017）
肌肉骨骼模型、18块（下肢体）肌肉

1. 随机肌肉激活
2. DDPG 基线
3. One of the early policies learnt by DDPG
4. Sample policy from TRPO baseline
5. Winning solution by @NNAISENSE


## FAQ
###### ImportError: ('Unable to load EGL library', "Could not find module 'EGL'
> `pip install glfw==2.6.4`
> 
> 使用 conda 安装 environment.yaml 中的 dependencies
> 
> `conda install -c conda-forge 指定版本`

## 参考

* [moychallenge-2024](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2024) - Manipulation and locomotion

* [myochallenge-2025](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2025) - Soccer Shootout and Table Tennis Rally, [eval.ai](https://eval.ai/web/challenges/challenge-page/2628/overview)

* [Existing locomotion task library based on MuJoCo](https://github.com/google-deepmind/dm_control/tree/main/dm_control/locomotion)

* [OpenSim 2019 第二名：Sample Efficient Ensemble Learning with Catalyst.RL](https://github.com/Scitator/run-skeleton-run-in-3d) - [视频讲解](https://www.youtube.com/watch?v=PprDcJHrFdg&t=4020s) ；其他解决方案：[第六名：learn_to_move](https://github.com/kamenbliznashki/learn_to_move)

* [假肢挑战赛](https://github.com/rwightman/pytorch-opensim-rl?tab=readme-ov-file) - 适用于 OpenSim 环境的 PyTorch 强化学习

* [Undergraduate thesis](https://github.com/OpenHUTB/sim/tree/master/pedestrian)