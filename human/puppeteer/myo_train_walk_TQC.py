# myo_finetune_stair_TQC.py
import gymnasium as gym
import os
import myosuite
import torch as th
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from sb3_contrib import TQC

# ==========================================
# 1. 配置路径 (区分预训练模型和新微调模型)
# ==========================================
# 预训练模型路径 (你之前在平地训练好的模型)
PRETRAINED_MODEL_PATH = "./policy/myowalk_tqc.zip"
PRETRAINED_VEC_PATH = "./policy/myowalk_tqc.pkl"

# 微调后新模型保存路径 (独立文件夹，防止覆盖)
LOG_DIR = "./logs/myorough_tqc/"
MODEL_DIR = "./policy/myorough_tqc/"
BEST_MODEL_DIR = "./best_model/myorough_tqc/"
VEC_NORM_PATH = os.path.join(MODEL_DIR, "myorough_tqc.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ==========================================
# 2. 创建环境
# ==========================================
def make_env(rank, seed=0):
    def _init():
        # [修改点 1]：明确指定为走楼梯环境
        env = gym.make('myoLegRoughTerrainWalk-v0')
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env

    return _init


# 自定义回调：定期保存新的 VecNormalize 统计数据
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            self.training_env.save(self.save_path)
        return True


# ----------------------------------------------------------------
# 程序入口
# ----------------------------------------------------------------
if __name__ == '__main__':

    # 检查预训练文件是否存在，防患于未然
    if not os.path.exists(PRETRAINED_MODEL_PATH) or not os.path.exists(PRETRAINED_VEC_PATH):
        raise FileNotFoundError(
            "找不到预训练的模型或环境统计量文件，请检查 PRETRAINED_MODEL_PATH 和 PRETRAINED_VEC_PATH。")

    num_cpu = 16
    print(f"正在启动 {num_cpu} 个并行环境准备进行楼梯微调 (基于 i9-13980HX)...")

    # ==========================================
    # 3. 核心修改：加载并同步 VecNormalize 统计量
    # ==========================================
    # 3.1 创建并行基础环境
    tmp_env = SubprocVecEnv([make_env(i, seed=42) for i in range(num_cpu)])

    # 3.2 [核心药引] 加载平地的统计量到楼梯环境中
    print(f"加载预训练环境统计量: {PRETRAINED_VEC_PATH}")
    env = VecNormalize.load(PRETRAINED_VEC_PATH, tmp_env)
    # 微调时必须开启训练模式，允许统计量根据新环境（楼梯的碰撞）进行微小的偏移校准
    env.training = True
    env.norm_reward = False

    # # 严格限制观测值的上下界，防止底层物理引擎爆炸产生的极端值传入网络
    # env.clip_obs = 10.0

    # ==========================================
    # 4. 评估环境 (Eval Env) 同步修改
    # ==========================================
    tmp_eval_env = DummyVecEnv([make_env(rank=999, seed=12345)])

    # Eval 环境也必须加载原本的统计量，但保持 training=False
    eval_env = VecNormalize.load(PRETRAINED_VEC_PATH, tmp_eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=BEST_MODEL_DIR,
                                 log_path='./results_myorough_tqc/',
                                 eval_freq=100000 // num_cpu,
                                 deterministic=True,
                                 render=False)

    save_vec_callback = SaveVecNormalizeCallback(save_path=VEC_NORM_PATH)
    callbacks = [eval_callback, save_vec_callback]

    # ==========================================
    # 5. 核心修改：加载预训练模型
    # ==========================================
    print(f"正在加载 TQC 预训练模型权重: {PRETRAINED_MODEL_PATH}")

    custom_objects = {
        # 稍微拉高一点学习率，给它冲出局部最优的动力（原本建议5e-5，现在给8e-5）
        "learning_rate": 8e-5,
        # 强制提高动作的随机性（熵），数值越大，它越容易“乱动”去尝试迈腿
        "ent_coef": "auto_0.05"
    }

    # 通过 TQC.load 恢复模型，并将新的 env 绑定给它
    model = TQC.load(PRETRAINED_MODEL_PATH, env=env, device="cuda", custom_objects=custom_objects)

    # 重新指定 Tensorboard 路径，方便和之前的训练对比
    model.tensorboard_log = "./tensorboard/tqc_myorough_tensorboard/"

    # # 强制让模型在新的楼梯环境中先随机探索或使用旧策略收集 10000 步数据填补 Buffer，再开始计算梯度
    # model.learning_starts = model.num_timesteps + 10000

    # ==========================================
    # 6. 开始微调训练
    # ==========================================
    print("开始在楼梯环境下进行课程学习微调 (Fine-tuning)...")
    try:
        # 楼梯环境可能需要较长的时间适应，先给 1000 万步
        # reset_num_timesteps=False 意味着 Tensorboard 会接着以前的步数继续画图
        model.learn(total_timesteps=20000000, callback=callbacks, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("手动停止微调训练，正在保存...")

    # 保存微调后的最终模型和环境统计量
    model.save(os.path.join(MODEL_DIR, "myorough_tqc_final"))
    env.save(VEC_NORM_PATH)
    print(f"微调完成！")
    print(f"模型已保存至 {MODEL_DIR}")
    print(f"环境统计量已保存至 {VEC_NORM_PATH}")