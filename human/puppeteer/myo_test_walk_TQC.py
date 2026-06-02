# test_and_record_myo.py
import os
import gymnasium as gym
import myosuite  # 必须导入以注册环境
import imageio
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC

# ==========================================
# 1. 配置路径 (需与训练代码保持完全一致)
# ==========================================
MODEL_DIR = "./policy/myostair_tqc/"
# MODEL_PATH = os.path.join(MODEL_DIR, "myostair_tqc_final.zip")

# 可以注释掉上面那行，改用 EvalCallback 保存的最佳模型：
MODEL_PATH = "./best_model/myostair_tqc/best_model.zip"

VEC_NORM_PATH = os.path.join(MODEL_DIR, "myostair_tqc.pkl")
VIDEO_FOLDER = "./videos/myowalk_tqc_videos/"
os.makedirs(VIDEO_FOLDER, exist_ok=True)
current_time = time.strftime("%Y%m%d-%H%M%S")
# 这是最终的 MP4 文件完整路径
video_path = os.path.join(VIDEO_FOLDER, f"walk_test_{current_time}.mp4")


def make_test_env():
    """创建单线程测试环境"""

    def _init():
        # MyoSuite 部分版本不支持 kwargs render_mode，如果继续报警告可以把 render_mode 去掉
        # 只要我们在下面用 fallback 渲染法，这里传不传都能拿到画面
        try:
            env = gym.make('myoLegStairTerrainWalk-v0', render_mode="rgb_array")
        except TypeError:
            env = gym.make('myoLegStairTerrainWalk-v0')
        return env

    return _init


if __name__ == '__main__':
    print("正在初始化测试环境...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")
    if not os.path.exists(VEC_NORM_PATH):
        raise FileNotFoundError(f"找不到归一化文件: {VEC_NORM_PATH}")

    # ==========================================
    # 2. 构建与恢复环境状态
    # ==========================================
    env = DummyVecEnv([make_test_env()])

    # 加载训练时的均值和方差，严禁测试时更新
    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False
    env.norm_reward = False

    # ==========================================
    # 3. 加载模型
    # ==========================================
    print(f"正在加载 TQC 模型: {MODEL_PATH}")
    model = TQC.load(MODEL_PATH, env=env, device="cuda")

    # ==========================================
    # 4. 运行评估并录制视频
    # ==========================================
    print("开始进行评估并录制视频...")

    obs = env.reset()
    frames = []

    MAX_STEPS = 1000
    episode_reward = 0.0

    for step in range(MAX_STEPS):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]

        # --- [核心修复：兼容 MyoSuite 的鲁棒渲染机制] ---
        frame = None
        try:
            # 尝试标准渲染
            frame_data = env.render()
            if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                frame = frame_data
        except Exception:
            pass

        # 如果标准渲染失败，启用底层 MuJoCo 离屏渲染 (Fallback)
        if frame is None:
            try:
                # 穿透 VecNormalize 和 DummyVecEnv 拿到最底层的 myosuite 环境
                base_env = env.venv.envs[0].unwrapped
                # 强制调用 MuJoCo 渲染 640x480 的画面
                frame = base_env.sim.renderer.render_offscreen(width=640, height=480, camera_id=0)
            except Exception as e:
                print(f"底层渲染也失败了，错误信息: {e}")
                pass

        # 确保只将有效的 NumPy 数组加入帧列表
        if frame is not None:
            # MuJoCo 渲染出来的图像可能是上下颠倒的，如果发现视频倒置，取消下一行的注释：
            # frame = np.flipud(frame)
            frames.append(frame)

        if step % 100 == 0:
            print(f"已处理 {step}/{MAX_STEPS} 步，捕获到 {len(frames)} 帧画面...")

        if done[0]:
            print(f"Agent 在第 {step} 步触发了 Done，当前 Episode 结束。")
            break

    print(f"测试完成！总奖励: {episode_reward:.2f}")

    # ==========================================
    # 5. 导出为 MP4 视频
    # ==========================================
    if len(frames) == 0:
        print("❌ 警告：未能捕获到任何画面帧，请检查环境渲染机制。")
    else:
        # [核心修复：传入具体的文件路径 video_path，而不是文件夹 VIDEO_FOLDER]
        print(f"正在合成视频，将保存至: {video_path}")
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
        print("🎉 视频生成完毕！可以打开 MP4 文件查看表现了。")