import imageio
import numpy as np
import gymnasium as gym
import myosuite
from stable_baselines3 import PPO
from datetime import datetime  # 引入时间处理库
import os  # 引入操作系统库，用于管理文件路径
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. 创建环境
# env = gym.make('myoLegWalk-v0')
env = DummyVecEnv([lambda: gym.make('myoLegWalk-v0')])

# 2. 加载归一化参数 (非常重要！否则模型就像没戴眼镜一样看不清)
# training=False 表示测试时不更新均值和方差
env = VecNormalize.load("vec_normalize.pkl", env)
env.training = False
env.norm_reward = False # 测试时不需要归一化奖励

# 3. 加载模型
model = PPO.load("my_walking_policy")
best_model = PPO.load("./best_model/best_model.zip")

# # --- 修改点 1: 处理 reset 的返回值 ---
# # 新版 gym 返回 (obs, info)，我们需要把 info 丢掉，只取 obs
# obs, info = env.reset()
#
# frames = []
#
# print("开始生成视频...")
# for i in range(300):
#     # 现在 obs 已经是纯数组了，predict 不会报错
#     action, _states = model.predict(obs, deterministic=True)
#     # action, _states = best_model.predict(obs, deterministic=True)
#
#     # --- 修改点: 处理 step 的返回值 ---
#     # 新版 step 返回 5 个值，旧版返回 4 个。这里做一个兼容判断。
#     step_result = env.step(action)
#
#     if len(step_result) == 5:
#         # 新版 API (Gymnasium / Gym >= 0.26)
#         obs, reward, terminated, truncated, info = step_result
#         done = terminated or truncated
#     else:
#         # 旧版 API
#         obs, reward, done, info = step_result
#
#     # 收集画面
#     rgb_img = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=0)
#     # 如果画面看起来是倒的，取消下面这行的注释
#     # rgb_img = np.flipud(rgb_img)
#
#     frames.append(rgb_img)
#
#     if done:
#         # --- 修改点 3: 这里也要改 ---
#         obs, info = env.reset()
#
# env.close()


# --- 修改点 1: VecEnv 的 reset 只返回 obs (SB3 的特性) ---
obs = env.reset()  # 不要写成 obs, info = ...

frames = []

print("开始生成视频...")
for i in range(300):
    action, _states = model.predict(obs, deterministic=True)
    # action, _states = best_model.predict(obs, deterministic=True)

    # --- 修改点 2: VecEnv 的 step 固定返回 4 个值 ---
    # VecEnv 会自动处理 reset，如果 done=True，obs 会自动变为新回合的初始帧
    obs, rewards, dones, infos = env.step(action)

    # --- 修改点 3: 访问底层仿真器 ---
    # 因为 env 被 DummyVecEnv 包裹了，需要通过 envs[0] 访问原始环境
    # 注意：确保这里只开了一个环境 (DummyVecEnv([lambda: ...]))
    raw_env = env.envs[0]
    rgb_img = raw_env.sim.renderer.render_offscreen(width=640, height=480, camera_id=0)

    # 如果画面看起来是倒的，取消下面这行的注释
    # rgb_img = np.flipud(rgb_img)

    frames.append(rgb_img)

    # VecEnv 会自动 reset，所以这里不需要手动 if done: env.reset()
    # 如果你想在第一次 done 时就停止录制，可以使用 break
    # if dones[0]:
    #     break

env.close()

# 3. 定义保存视频的文件夹名称
save_folder = "myo_training_videos"

# 4. 检查文件夹是否存在，不存在则创建
# exist_ok=True 表示如果文件夹已经存在，不会报错，直接继续
os.makedirs(save_folder, exist_ok=True)

# 5. 生成文件名和完整路径
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f'trained_walk_{current_time}.mp4'

# 使用 os.path.join 拼接文件夹和文件名 (兼容 Windows/Linux)
full_path = os.path.join(save_folder, video_filename)

# 6. 保存视频到指定路径
imageio.mimsave(full_path, frames, fps=30)
print(f"视频已保存至：{full_path}")