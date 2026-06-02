import argparse
import os
import time
import gymnasium as gym
import myosuite
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC
import imageio

# 1. 算法映射（与训练脚本保持一致）
ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TQC": TQC
}


def test(args):
    # --- 1. 路径构建 ---
    # 根据训练时的命名规则反推路径：包含 env 和 mode_str
    mode_str = "finetune" if args.finetune else "train"
    run_name = f"{args.algo}_{args.env}_{mode_str}_seed{args.seed}"
    base_dir = "./logs/logs_comparison"
    video_folder = f"./videos/{run_name}_videos/"
    os.makedirs(video_folder, exist_ok=True)

    # 默认模型路径
    if args.model_path:
        model_path = args.model_path
        # 假设 vec_normalize.pkl 在模型同一目录下
        vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    else:
        model_dir = f"{base_dir}/models/{run_name}/"
        # # 优先加载 final_model，如果不存在则尝试 best_model
        # if os.path.exists(os.path.join(model_dir, "final_model.zip")):
        #     model_path = os.path.join(model_dir, "final_model")
        # else:
        print(f"Warning: Final model not found in {model_dir}, trying best_model...")
        model_path = f"{base_dir}/best_model/{run_name}/best_model"
        vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")

    print(f"Loading Model from: {model_path}")
    print(f"Loading VecNormalize from: {vec_norm_path}")

    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"找不到归一化文件: {vec_norm_path}。没有它模型无法正常工作！")

    # --- 2. 创建环境 ---
    # 基础环境不传 render_mode，避免 MyoSuite 报错 Unused kwargs
    def make_test_env():
        def _init():
            # MyoSuite 部分版本不支持 kwargs render_mode，使用 Try-Except 优雅解决
            try:
                r_mode = "rgb_array" if args.save_video else "human"
                env = gym.make(args.env, render_mode=r_mode)
            except TypeError:
                env = gym.make(args.env)
            return env
        return _init

    # 包装进 DummyVecEnv
    env = DummyVecEnv([make_test_env()])

    # --- 3. [核心] 加载归一化统计数据 ---
    # 我们必须告诉测试环境：训练时的数据均值和方差是多少
    # training=False: 测试时不要更新均值方差，只读取！
    # norm_reward=False: 测试时我们想看真实的奖励分数，不要归一化奖励
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False

    # --- 4. 加载模型 ---
    AlgoClass = ALGO_MAP[args.algo]
    model = AlgoClass.load(model_path, env=env, device="auto")

    # --- 5. 开始循环测试 ---
    print("\nStarting Testing... Press Ctrl+C to stop.")

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        # 准备一个空列表，用来装这一局的所有画面帧
        frames = []

        while not done:
            # deterministic=True: 测试时通常使用确定性策略（不加随机噪声），表现更稳
            # 除非你想测试模型的鲁棒性，否则建议设为 True
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            if args.save_video:
                frame = None
                try:
                    # 尝试标准渲染
                    frame_data = env.render()
                    if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                        frame = frame_data
                except Exception:
                    pass

                if frame is None:
                    try:
                        # 穿透 VecNormalize (venv) 和 DummyVecEnv 拿到最底层的 myosuite 环境
                        base_env = env.venv.envs[0].unwrapped
                        # 强制调用 MuJoCo 渲染 640x480 的画面
                        frame = base_env.sim.renderer.render_offscreen(width=640, height=480, camera_id=0)
                    except Exception as e:
                        pass
                if frame is not None:
                    # 如果你发现录出来的视频是上下颠倒的，取消下面这行的注释即可：
                    # frame = np.flipud(frame)
                    frames.append(frame)

            else:
                # 实时窗口预览（同样加入防崩溃逻辑）
                try:
                    base_env = env.venv.envs[0].unwrapped
                    base_env.sim.renderer.render_to_window()
                except Exception:
                    env.render()

            # 防止死循环（有些环境 done 信号有问题）
            if args.max_steps and step > args.max_steps:
                print("Max steps reached.")
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward[0]:.2f}, Steps = {step}")
        if args.save_video and len(frames) > 0:
            # 引入时间戳，防止多次跑测试时覆盖之前的视频
            current_time = time.strftime("%Y%m%d-%H%M%S")
            video_name = f"{args.env}_ep{episode + 1}_{current_time}.mp4"
            video_path = os.path.join(video_folder, video_name)

            print(f"正在合成视频，将保存至: {video_path} (共 {len(frames)} 帧)...")
            # macro_block_size=1 是极佳的实践，防止 imageio 在处理分辨率不是 16 的倍数时报错
            imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
            print("🎉 视频生成完毕！可以打开 MP4 文件查看表现了。")
        elif args.save_video:
            print("❌ 警告：未能捕获到任何画面帧，请检查环境渲染机制。")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["PPO", "SAC", "TQC"], help="Algorithm used")
    parser.add_argument("--env", type=str, default="myoLegWalk-v0", help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Seed used during training")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to test")
    parser.add_argument("--model_path", type=str, default=None, help="Optional: Manually specify model path")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--save_video", action="store_true", help="是否将测试过程保存为 mp4 视频")

    # 新增：为了与训练脚本对齐，增加微调模式的标识
    parser.add_argument("--finetune", action="store_true", help="如果测试的是微调(finetune)生成的模型，请加上此参数")

    args = parser.parse_args()
    test(args)