import argparse
import os
import gymnasium as gym
import myosuite
import numpy as np
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from sb3_contrib import TQC

# ===========================
# 1. 全局配置与参数定义
# ===========================

# 定义不同算法的默认超参数 (基于你提供的文件整理)
HYPERPARAMS = {
    "PPO": {
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 3e-4,
            "batch_size": 256,  # PPO 默认通常较小，可调整
        }
    },
    "SAC": {
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 3e-4,
            "batch_size": 1024,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_steps": 1,
            "train_freq": 1,
            "use_sde": True,
            "use_sde_at_warmup": True,
            "policy_kwargs": dict(net_arch=[400, 300], log_std_init=-2, use_sde=True),
        }
    },
    "TQC": {
        "policy": "MlpPolicy",
        "kwargs": {
            "learning_rate": 3e-4,
            "batch_size": 1024,
            "top_quantiles_to_drop_per_net": 2,
            "use_sde": True,
            "policy_kwargs": dict(net_arch=[512, 512], n_critics=2, use_sde=True, log_std_init=-2),
            "device": "cuda"
        }
    }
}

ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TQC": TQC
}


class SaveVecNormalizeCallback(BaseCallback):
    """用于定期保存 VecNormalize 统计数据的回调"""

    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            self.training_env.save(self.save_path)
        return True


def make_env(env_id, rank, seed=0):
    """创建环境的工厂函数"""

    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)  # 关键：不同进程使用不同随机种子
        env = Monitor(env)
        return env

    return _init


def train(args):
    # --- 1. 路径设置 ---
    run_name = f"{args.algo}_seed{args.seed}"  # 实验名称
    base_dir = "./logs/logs_comparison"
    tensorboard_log = f"{base_dir}/tensorboard/"
    model_save_dir = f"{base_dir}/models/{run_name}/"
    best_model_dir = f"{base_dir}/best_model/{run_name}/"
    vec_norm_path = os.path.join(model_save_dir, "vec_normalize.pkl")

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print(f"Set up: Algo={args.algo}, Workers={args.num_cpu}, Seed={args.seed}")

    # --- 2. 创建训练环境 ---
    # 如果 num_cpu > 1，使用多进程 SubprocVecEnv，否则使用 DummyVecEnv
    env_factory = [make_env(args.env, i, args.seed) for i in range(args.num_cpu)]

    if args.num_cpu > 1:
        env = SubprocVecEnv(env_factory)
    else:
        env = DummyVecEnv(env_factory)

    # 归一化处理 (所有算法通用)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # --- 3. 创建评估环境 ---
    # 评估环境只需一个进程，且 norm_reward=False (我们要看真实奖励)
    eval_env = DummyVecEnv([make_env(args.env, 999, args.seed)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=0.99, training=False)

    # --- 4. 回调函数 ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=max(10000 // args.num_cpu, 1),  # 自动调整频率
        deterministic=True,
        render=False
    )
    save_vec_callback = SaveVecNormalizeCallback(save_path=vec_norm_path)
    callbacks = [eval_callback, save_vec_callback]

    # --- 5. 初始化模型 ---
    AlgoClass = ALGO_MAP[args.algo]
    hyperparams = HYPERPARAMS[args.algo]

    model = AlgoClass(
        hyperparams["policy"],
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        **hyperparams["kwargs"]  # 展开对应算法的参数
    )

    # --- 6. 开始训练 ---
    print(f"Starting training for {args.total_timesteps} steps...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, tb_log_name=run_name)
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    # --- 7. 保存最终结果 ---
    model.save(os.path.join(model_save_dir, "final_model"))
    env.save(vec_norm_path)
    print(f"Model saved to {model_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified RL Training Script")

    # 核心参数
    parser.add_argument("--algo", type=str, required=True, choices=["PPO", "SAC", "TQC"], help="Algorithm to use")
    parser.add_argument("--env", type=str, default="myoLegWalk-v0", help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_cpu", type=int, default=1, help="Number of parallel CPU workers")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000, help="Total training timesteps")

    args = parser.parse_args()

    train(args)