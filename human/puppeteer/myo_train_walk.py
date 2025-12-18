import gym
import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. 创建环境
# 使用 DummyVecEnv 包装环境，这是 SB3 要求的格式
env = DummyVecEnv([lambda: gym.make('myoLegWalk-v0')])

# 2. 定义模型 (大脑)
# PPO 是一种非常稳健的强化学习算法
# MlpPolicy 表示使用多层感知机 (神经网络)
# verbose=1 会在终端打印训练进度
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_walk_tensorboard/")

print("开始训练... (这可能需要很久，取决于步数)")

# 3. 开始训练
# total_timesteps 是训练的总步数。
# 想要真正学会走路，通常需要 1,000,000 (一百万) 步以上。
# 为了演示代码是否跑通，这里先设为 10,000 (效果不会好，但能跑完)
model.learn(total_timesteps=1000000)

# 4. 保存模型
model.save("my_walking_policy")
print("训练完成，模型已保存为 my_walking_policy.zip")