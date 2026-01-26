import gymnasium as gym
import os
import myosuite
import torch as th  # NEW: 用于优化器配置
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # NEW: 引入归一化
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from sb3_contrib import TQC

# 1. 配置
LOG_DIR = "./logs/myo_logs/"
MODEL_DIR = "./policy/"
BEST_MODEL_DIR = "./best_model/best_model_sac/"
VEC_NORM_PATH = os.path.join(MODEL_DIR, "vec_normalize_sac.pkl") # NEW: 保存归一化参数的路径

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# 2. 创建环境 (这是最关键的一步变化)
def make_env():
    env = gym.make('myoLegWalk-v0')
    env = Monitor(env) # 记录数据
    return env

# 创建向量化环境
env = DummyVecEnv([make_env])

# [NEW - 核心药引]: 使用 VecNormalize 对观测值(Obs)和奖励(Reward)进行归一化
# norm_obs=True: 计算观测值的均值和方差，将输入标准化到均值为0、方差为1的分布。这对神经网络收敛至关重要。
# norm_reward=True: 缩放奖励值。这有助于稳定梯度的幅度，防止奖励过大导致训练不稳定。
# gamma=0.99: 折扣因子，归一化奖励时需要用到，必须与模型训练时的 gamma 保持一致。
# clip_obs=10.: 将观测值截断在 [-10, 10] 之间，防止极端异常值破坏网络权重。
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

# 3. 定义 SAC 模型
# [NEW - 平滑剂]: log_std_init=-2 让初始探索更谨慎，配合 gSDE
policy_kwargs = dict(
    net_arch=[400, 300],
    use_sde=True,        # [NEW] 开启 gSDE (状态依赖探索)，让动作极其平滑
    log_std_init=-2,     # [NEW] 初始探索噪声降低，减少剧烈抽搐
)

# 自定义回调：定期保存 VecNormalize 的统计数据
# 如果不保存这个，测试时模型就是“瞎子”
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            self.model.get_env().save(self.save_path) # 保存环境统计量(均值方差)
        return True

save_vec_callback = SaveVecNormalizeCallback(save_path=VEC_NORM_PATH)

# 定义回调函数
eval_env = DummyVecEnv([make_env])
# 注意：Eval 环境通常不进行训练时的 norm_reward 更新，但需要使用训练集的 norm_obs 参数
# 这里简化处理，Eval 时暂时不 Normalize，或者你可以再创建一个 VecNormalize 并加载训练集的 stats
# 为了简单起见，我们主要看 Tensorboard，Eval 留作基本检查

# [修复核心]: 必须给 eval_env 也加上 VecNormalize
# norm_reward=False: 验证时我们要看真实的奖励值，不要归一化
# training=False: 验证时不要更新均值和方差，只使用训练集同步过来的参数
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=0.99, training=False)
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=BEST_MODEL_DIR,
                             log_path='./results_sac/',
                             eval_freq=10000,
                             deterministic=True,
                             render=False)

# 组合回调
callbacks = [eval_callback, save_vec_callback]

print("开始 SAC (gSDE + VecNormalize) 训练...")
# [NEW]: 调整参数
# learning_rate: 使用线性衰减或更保守的值 1e-4，这里先保持 3e-4
# batch_size: 1024 (原256) -> 在复杂物理环境中，大 Batch 能让梯度方向更准
# use_sde: True -> 关键！平滑动作
# model = SAC("MlpPolicy",
#             env,
#             verbose=1,
#             tensorboard_log="./tensorboard/sac_walk_tensorboard/",
#             learning_rate=3e-4,
#             buffer_size=1000000,
#             batch_size=1024,      # [NEW] 更大 Batch
#             tau=0.005,            # [NEW] 软更新系数，0.005 比默认值更稳
#             gamma=0.99,           # 关注长远奖励
#             train_freq=1,
#             gradient_steps=1,
#             use_sde=True,         # [NEW] 开启平滑探索
#             use_sde_at_warmup=True, # [NEW] 预热阶段也使用平滑探索
#             policy_kwargs=policy_kwargs)

# TQC 通常使用 "top_quantiles_to_drop_per_net" 参数来控制保守程度
model = TQC("MlpPolicy",
            env,
            top_quantiles_to_drop_per_net=2, # 关键参数：截断头部过高估值
            verbose=1,
            batch_size=1024, # 保持大 Batch
            learning_rate=3e-4,
            use_sde=True,    # 依然开启 gSDE
            policy_kwargs=dict(net_arch=[512, 512], n_critics=2), # 加宽网络
            tensorboard_log="./tensorboard/tqc_walk_tensorboard/"
            )

# 训练步数建议增加，肌骨骼很难训练
# 先跑 2M 步看看效果
try:
    model.learn(total_timesteps=10000000, callback=callbacks)
except KeyboardInterrupt:
    print("手动停止训练，正在保存...")

# 4. 保存最终模型和环境统计量
model.save(os.path.join(MODEL_DIR, "./policy/my_walking_policy_sac"))
env.save(VEC_NORM_PATH) # [重要] 必须保存环境统计量，否则测试时无法还原视角
print(f"SAC 训练完成。模型已保存至 {MODEL_DIR}")
print(f"环境统计量已保存至 {VEC_NORM_PATH} (测试脚本必须加载此文件!)")