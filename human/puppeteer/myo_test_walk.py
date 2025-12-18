import imageio
import numpy as np
import gym
import myosuite
from stable_baselines3 import PPO

# 1. 创建环境
env = gym.make('myoLegWalk-v0')

# 2. 加载模型
model = PPO.load("my_walking_policy")

# --- 修改点 1: 处理 reset 的返回值 ---
# 新版 gym 返回 (obs, info)，我们需要把 info 丢掉，只取 obs
obs, info = env.reset()

frames = []

print("开始生成视频...")
for i in range(300):
    # 现在 obs 已经是纯数组了，predict 不会报错
    action, _states = model.predict(obs, deterministic=True)

    # --- 修改点 2: 处理 step 的返回值 ---
    # 新版 step 返回 5 个值，旧版返回 4 个。这里做一个兼容判断。
    step_result = env.step(action)

    if len(step_result) == 5:
        # 新版 API (Gymnasium / Gym >= 0.26)
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        # 旧版 API
        obs, reward, done, info = step_result

    # 收集画面
    rgb_img = env.sim.renderer.render_offscreen(width=640, height=480, camera_id=0)
    # 如果画面看起来是倒的，取消下面这行的注释
    # rgb_img = np.flipud(rgb_img)

    frames.append(rgb_img)

    if done:
        # --- 修改点 3: 这里也要改 ---
        obs, info = env.reset()

env.close()

# 保存视频
imageio.mimsave('trained_walk.mp4', frames, fps=30)
print("视频已保存：trained_walk.mp4")