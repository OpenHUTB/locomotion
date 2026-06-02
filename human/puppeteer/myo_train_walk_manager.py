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
    # 如果是微调模式，我们可以在 run_name 中加入 finetune 标识，方便在 Tensorboard 中对比
    mode_str = "finetune" if args.finetune else "train"
    run_name = f"{args.algo}_{args.env}_{mode_str}_seed{args.seed}"  # 实验名称
    base_dir = "./logs/logs_comparison"
    tensorboard_log = f"{base_dir}/tensorboard/"
    model_save_dir = f"{base_dir}/models/{run_name}/"
    best_model_dir = f"{base_dir}/best_model/{run_name}/"
    vec_norm_path = os.path.join(model_save_dir, "vec_normalize.pkl")

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print(f"Set up: Algo={args.algo}, Env={args.env}, Mode={mode_str}, Workers={args.num_cpu}, Seed={args.seed}")

    # --- 2. 创建训练环境 ---
    # 如果 num_cpu > 1，使用多进程 SubprocVecEnv，否则使用 DummyVecEnv
    env_factory = [make_env(args.env, i, args.seed) for i in range(args.num_cpu)]

    if args.num_cpu > 1:
        env = SubprocVecEnv(env_factory)
    else:
        env = DummyVecEnv(env_factory)

    # 归一化处理 (所有算法通用)
    if args.finetune and args.pretrained_vec_path:
        if not os.path.exists(args.pretrained_vec_path):
            raise FileNotFoundError(f"找不到预训练的环境统计量文件: {args.pretrained_vec_path}")
        print(f"Loading pretrained VecNormalize from {args.pretrained_vec_path}")
        env = VecNormalize.load(args.pretrained_vec_path, env)
        env.training = True
        env.norm_reward = False  # 微调时通常不归一化 reward，看真实奖励
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # --- 3. 创建评估环境 ---
    # 评估环境只需一个进程，且 norm_reward=False (我们要看真实奖励)
    eval_env = DummyVecEnv([make_env(args.env, 999, args.seed)])

    # 同步评估环境的归一化逻辑
    if args.finetune and args.pretrained_vec_path:
        eval_env = VecNormalize.load(args.pretrained_vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
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

    # 模型的条件初始化与 Custom Objects 注入
    if args.finetune and args.pretrained_model_path:
        if not os.path.exists(args.pretrained_model_path):
            raise FileNotFoundError(f"找不到预训练的模型文件: {args.pretrained_model_path}")

        print(f"Loading pretrained model from {args.pretrained_model_path}")

        # 动态构建 custom_objects
        custom_objects = {}
        if args.ft_learning_rate is not None:
            custom_objects["learning_rate"] = args.ft_learning_rate
        if args.ft_ent_coef is not None:
            # 如果输入的是纯数字字符串(如 "0.05")，就转成 float；如果是带字母的(如 "auto_0.05")，就保持原样
            try:
                custom_objects["ent_coef"] = float(args.ft_ent_coef)
            except ValueError:
                custom_objects["ent_coef"] = args.ft_ent_coef

        model = AlgoClass.load(
            args.pretrained_model_path,
            env=env,
            device=hyperparams["kwargs"].get("device", "auto"),
            custom_objects=custom_objects
        )
        # 必须重新指定 tensorboard 路径，否则不会记录
        model.tensorboard_log = tensorboard_log
    else:
        model = AlgoClass(
            hyperparams["policy"],
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            **hyperparams["kwargs"]
        )

    # --- 6. 开始训练 ---
    print(f"Starting training for {args.total_timesteps} steps...")
    try:
        # 微调时保持 Tensorboard 步数连续
        reset_timesteps = not args.finetune
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("Training interrupted manually.")

    # --- 7. 保存最终结果 ---
    model.save(os.path.join(model_save_dir, "final_model"))
    env.save(vec_norm_path)
    print(f"Model saved to {model_save_dir}")
    print(f"VecNormalize stats saved to {vec_norm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified RL Training and Fine-tuning Script")

    # 核心参数
    parser.add_argument("--algo", type=str, required=True, choices=["PPO", "SAC", "TQC"], help="Algorithm to use")
    parser.add_argument("--env", type=str, default="myoLegWalk-v0", help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_cpu", type=int, default=1, help="Number of parallel CPU workers")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000, help="Total training timesteps")

    # 微调专属参数
    parser.add_argument("--finetune", action="store_true", help="启用微调模式")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="预训练模型 .zip 的路径")
    parser.add_argument("--pretrained_vec_path", type=str, default=None, help="预训练环境统计量 .pkl 的路径")
    parser.add_argument("--ft_learning_rate", type=float, default=None, help="微调时的覆盖学习率 (如 8e-5)")
    parser.add_argument("--ft_ent_coef", type=str, default=None, help="微调时的覆盖熵系数 (如 'auto_0.05')")

    args = parser.parse_args()
    train(args)