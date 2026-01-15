import gymnasium as gym
import myosuite
import numpy as np
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def evaluate_task(task_name, model_path=None, stats_path=None, num_episodes=5):
    """
    评测单个任务的性能。
    如果模型与环境不兼容（观测空间形状不同），自动降级为随机策略。
    """
    print(f"\n[开始评测] 任务: {task_name}")

    # 1. 创建环境
    try:
        raw_env = gym.make(task_name)
    except Exception as e:
        print(f"无法加载任务 {task_name}: {e}")
        return None

    # 使用 DummyVecEnv 包装，保持与训练时一致的接口
    env = DummyVecEnv([lambda: raw_env])

    # 2. 尝试加载训练好的模型和归一化参数
    model = None
    use_random_policy = True

    if model_path and os.path.exists(model_path) and stats_path and os.path.exists(stats_path):
        try:
            # 尝试加载归一化统计数据
            # 关键修改：VecNormalize.load 会检查 obs shape，如果不匹配会抛出 AssertionError
            # 我们捕获这个错误，从而实现“不匹配就跳过模型，只跑随机”
            loaded_env = VecNormalize.load(stats_path, env)
            loaded_env.training = False  # 测试模式，不更新均值方差
            loaded_env.norm_reward = False

            # 如果上面没报错，说明形状匹配，可以将 env 替换为带归一化的版本
            env = loaded_env

            # 加载策略网络
            model = PPO.load(model_path)
            print(f"✅ 成功加载模型: {model_path} (适用于 {task_name})")
            use_random_policy = False

        except (AssertionError, ValueError, RuntimeError) as e:
            # 捕获形状不匹配等错误
            print(f"⚠️  警告: 模型/统计数据与当前任务 {task_name} 不匹配。")
            print(f"    原因: 观测空间形状不同 (训练环境 vs 当前环境)。")
            print(f"    操作: 切换到随机策略运行 (仅测试环境流程)。")
            # 此时 env 仍然是普通的 DummyVecEnv，没有被 VecNormalize 覆盖，可以正常跑随机
            model = None
    else:
        print("ℹ️  未提供模型路径或文件不存在，使用随机策略运行。")

    # 3. 运行评测循环
    all_episode_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            if not use_random_policy and model is not None:
                # 使用训练好的策略
                action, _states = model.predict(obs, deterministic=True)
            else:
                # 使用随机策略 (跑通流程用)
                # VecEnv 的 step 需要接收 (n_envs, action_dim) 的输入
                action = [env.action_space.sample()]

            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            done = dones[0]

        all_episode_rewards.append(total_reward)
        print(f"  Episode {ep + 1}: Reward = {total_reward:.2f}")

    avg_reward = np.mean(all_episode_rewards)
    print(f"任务 {task_name} 平均奖励: {avg_reward:.2f}")
    return avg_reward


# ================= 配置区域 =================
if __name__ == "__main__":
    # 1. 定义你要测试的任务清单
    task_list = [
        'myoLegWalk-v0',  # 你的模型应该是在这个任务上训练的
        # 'myoHandReachFixed-v0',  # 观测维度不同，将自动降级为随机策略
        # 'myoElbowPose1D6MFixed-v0',  # 观测维度不同，将自动降级为随机策略
        'myoLegHillyTerrainWalk-v0',
        'myoLegStairTerrainWalk-v0',
        'myoLegRoughTerrainWalk-v0',
    ]

    # 2. 指定模型路径
    MODEL_PATH = "./best_model/best_model/best_model.zip"
    STATS_PATH = "vec_normalize.pkl"

    results = {}

    print("=" * 50)
    print("启动多任务评测脚本")
    print("=" * 50)

    for task in task_list:
        res = evaluate_task(task, MODEL_PATH, STATS_PATH, num_episodes=3)
        results[task] = res

    # 4. 打印最终汇总报告
    print("\n" + "=" * 40)
    print("      多任务性能评估汇总")
    print("=" * 40)
    for task, score in results.items():
        score_str = f"{score:.2f}" if score is not None else "失败"
        print(f"{task:<30} : {score_str}")
    print("=" * 40)