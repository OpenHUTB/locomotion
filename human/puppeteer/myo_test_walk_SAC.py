import gymnasium as gym
import myosuite
import os
import time
import numpy as np
import imageio
from stable_baselines3 import SAC
# 修改 myo_test_walk_SAC.py 的环境加载部分
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. 路径配置
# MODEL_PATH = "./policy/my_walking_policy_sac.zip"
MODEL_PATH = "./best_model/best_model_sac/best_model.zip"
VIDEO_FOLDER = "./videos/myowalk_sac_videos/"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
current_time = time.strftime("%Y%m%d-%H%M%S")
video_path = os.path.join(VIDEO_FOLDER, f"walk_test_{current_time}.mp4")

# 2. 创建环境
# 使用 rgb_array 模式初始化，尽管 RecordVideo 可能失效，但它能确保环境准备好图像渲染
env = gym.make('myoLegWalk-v0', render_mode='rgb_array')
env = DummyVecEnv([lambda: env]) # 必须包装成 VecEnv

# 2. 加载统计量 (这就是那副眼镜)
VEC_NORM_PATH = "./policy/vec_normalize_sac.pkl" # 训练脚本中保存的路径
if os.path.exists(VEC_NORM_PATH):
    print(f"正在加载环境统计量: {VEC_NORM_PATH}")
    env = VecNormalize.load(VEC_NORM_PATH, env)
    # 测试时不要更新统计量，也不要归一化 Reward
    env.training = False
    env.norm_reward = False
else:
    print("警告：未找到 vec_normalize.pkl，如果训练时使用了 VecNormalize，模型将无法正常工作！")

# 3. 加载模型
if os.path.exists(MODEL_PATH):
    print(f"正在加载模型: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)
else:
    print(f"错误: 找不到模型文件 {MODEL_PATH}")
    exit()

# 4. 运行测试并手动捕获帧
print(f"开始测试并记录视频...")
obs = env.reset()

frames = []
total_reward = 0
done = False
truncated = False

# 限制录制步数
max_steps = 4000
step_count = 0

while not (done or truncated) and step_count < max_steps:
    # SAC 推理
    action, _states = model.predict(obs, deterministic=True)
    # ... (前面的 model.predict 和 env.step 保持不变) ...
    obs, rewards, dones, infos = env.step(action)

    reward = rewards[0]
    done = dones[0]

    # --- 修复后的渲染逻辑 ---
    frame = None

    # 1. 尝试标准渲染 (虽然已知大概率会返回 None)
    try:
        frame = env.render()
    except Exception:
        pass  # 忽略报错，准备 fallback

    # 2. 如果标准渲染无效（返回 None 或失败），强制使用底层 Mujoco 渲染器
    if frame is None:
        try:
            # get_attr("sim") 会返回一个列表（每个环境一个 sim），我们取第一个
            # 这种方式可以自动穿透 VecNormalize, DummyVecEnv 等所有包装器
            sim = env.get_attr("sim")[0]

            # 直接调用 Mujoco 的渲染器
            # 注意：如果 'side_view' 报错，尝试改成 camera_id=0 或 -1 (free camera)
            frame = sim.renderer.render_offscreen(
                width=640,
                height=480,
                camera_id='side_view'
            )

            # 如果 render_offscreen 返回的是翻转的或者格式不对，有时需要 flip
            # frame = np.flipud(frame)

        except Exception as e:
            if step_count % 100 == 0:  # 避免刷屏
                print(f"底层渲染失败: {e}")

    # 3. 保存帧
    if frame is not None:
        frame = np.asarray(frame)
        frames.append(frame)

    total_reward += reward
    step_count += 1

    # 如果遇到 done，VecEnv 已经自动 reset 了，如果你只想录制一条轨迹，可以在这里 break
    if done:
        print(f"Episode finished at step {step_count}")
        break

# 5. 保存视频
if len(frames) > 0:
    print(f"正在保存视频到: {video_path}")
    # 确保 fps 与仿真频率匹配（通常 MyoSuite 步长较大，30-50fps 比较自然）
    imageio.mimsave(video_path, frames, fps=30)
    print(f"视频录制成功！共 {len(frames)} 帧。")
else:
    print("警告：未捕获到任何画面帧，请检查环境配置。")

print(f"测试结束！总步数: {step_count}, 总奖励: {total_reward:.2f}")
env.close()