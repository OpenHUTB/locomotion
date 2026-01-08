import gym
import os
import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor  # <--- 新增这行
from myo_dm_adapter import MyoDmAdapter
from stable_baselines3.common.callbacks import EvalCallback

# 1. 配置参数
ENV_NAME = 'myoLegWalk-v0'
LOG_DIR = "./myo_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

# 1. 创建环境
# 使用 DummyVecEnv 包装环境，这是 SB3 要求的格式
# env = DummyVecEnv([lambda: gym.make('myoLegWalk-v0')])
env = DummyVecEnv([lambda: Monitor(gym.make('myoLegWalk-v0'))])
# 添加 VecNormalize，但关闭训练模式和奖励归一化
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 2. 定义模型 (大脑)
# PPO 是一种非常稳健的强化学习算法
# MlpPolicy 表示使用多层感知机 (神经网络)
# verbose=1 会在终端打印训练进度
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_walk_tensorboard/")


# ----------------保存最好的模型----------------
# 1. 创建一个独立的评估环境（考试专用，不带随机噪声）
eval_env = DummyVecEnv([lambda: Monitor(gym.make('myoLegWalk-v0'))])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 2. 定义回调函数：每隔 10000 步测试一次，保存最好的模型到 ./best_model/ 文件夹
eval_callback = EvalCallback(eval_env,
                             best_model_save_path='./best_model/',
                             log_path='./results/',
                             eval_freq=10000,
                             deterministic=True,
                             render=False)
# --------------------------------------------

print("开始训练... (这可能需要很久，取决于步数)")

# 3. 开始训练
# total_timesteps 是训练的总步数。
# 想要真正学会走路，通常需要 1,000,000 (一百万) 步以上。
# 为了演示代码是否跑通，这里先设为 10,000 (效果不会好，但能跑完)
model.learn(total_timesteps=10000000, callback=eval_callback)

# 4. 保存模型
model.save("my_walking_policy")
env.save("vec_normalize.pkl") # 保存环境统计数据
print("训练完成，模型已保存为 my_walking_policy.zip")

