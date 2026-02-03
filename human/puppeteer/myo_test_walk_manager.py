import argparse
import os
import gymnasium as gym
import myosuite
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC

# 1. 算法映射（与训练脚本保持一致）
ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TQC": TQC
}


def test(args):
    # --- 1. 路径构建 ---
    # 根据训练时的命名规则反推路径
    # 如果你手动指定了完整路径 (--model_path)，则优先使用
    run_name = f"{args.algo}_seed{args.seed}"
    base_dir = "./logs/logs_comparison"

    # 默认模型路径
    if args.model_path:
        model_path = args.model_path
        # 假设 vec_normalize.pkl 在模型同一目录下
        vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    else:
        # 自动推断路径 (根据 train_manager.py 的保存逻辑)
        model_dir = f"{base_dir}/models/{run_name}/"
        # 优先加载 final_model，如果不存在则尝试 best_model
        if os.path.exists(os.path.join(model_dir, "final_model.zip")):
            model_path = os.path.join(model_dir, "final_model")
        else:
            print(f"Warning: Final model not found in {model_dir}, trying best_model...")
            model_path = f"{base_dir}/best_model/{run_name}/best_model"

        vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")

    print(f"Loading Model from: {model_path}")
    print(f"Loading VecNormalize from: {vec_norm_path}")

    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"找不到归一化文件: {vec_norm_path}。没有它模型无法正常工作！")

    # --- 2. 创建环境 ---
    # render_mode='human' 让我们可以看到动画窗口
    env = gym.make(args.env, render_mode='human')

    # 包装进 DummyVecEnv (SB3 模型要求输入必须是向量化环境)
    env = DummyVecEnv([lambda: env])

    # --- 3. [核心] 加载归一化统计数据 ---
    # 我们必须告诉测试环境：训练时的数据均值和方差是多少
    # training=False: 测试时不要更新均值方差，只读取！
    # norm_reward=False: 测试时我们想看真实的奖励分数，不要归一化奖励
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False

    # --- 4. 加载模型 ---
    AlgoClass = ALGO_MAP[args.algo]
    model = AlgoClass.load(model_path, env=env)

    # --- 5. 开始循环测试 ---
    print("\nStarting Testing... Press Ctrl+C to stop.")

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # deterministic=True: 测试时通常使用确定性策略（不加随机噪声），表现更稳
            # 除非你想测试模型的鲁棒性，否则建议设为 True
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            # 如果 render_mode='human' 生效，gym 会自动弹窗
            env.render()

            # 防止死循环（有些环境 done 信号有问题）
            if args.max_steps and step > args.max_steps:
                print("Max steps reached.")
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward[0]:.2f}, Steps = {step}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["PPO", "SAC", "TQC"], help="Algorithm used")
    parser.add_argument("--env", type=str, default="myoLegWalk-v0", help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Seed used during training")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to test")
    parser.add_argument("--model_path", type=str, default=None, help="Optional: Manually specify model path")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max steps per episode")

    args = parser.parse_args()
    test(args)