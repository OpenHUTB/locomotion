#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.*")
import numpy as np
import tensorflow as tf
import sonnet as snt
from myo_dm_adapter import MyoDmAdapter
from acme.wrappers import SinglePrecisionWrapper
from acme.agents.tf import ddpg
from acme import specs, EnvironmentLoop
from acme.utils import loggers
from acme.tf import networks as netlib
from acme.tf import utils as tf_utils
# 限制显存（可选）
gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# ---------- 自定义裁剪模块 ----------
class ClipByActSpec(snt.Module):
    def __init__(self, low, high, name=None):
        super().__init__(name=name)
        self._low  = tf.constant(low,  dtype=tf.float32)
        self._high = tf.constant(high, dtype=tf.float32)

    def __call__(self, x):
        return tf.clip_by_value(x, self._low, self._high)
# ------------------------------------

def main():
    # 1. 环境
    raw_env = MyoDmAdapter("myoElbowPose1D6MRandom-v0")
    act_low  = raw_env.gym.action_space.low.astype(np.float32)
    act_high = raw_env.gym.action_space.high.astype(np.float32)

    env  = SinglePrecisionWrapper(raw_env)
    spec = specs.make_environment_spec(env)

    # 2. 手工网络（带裁剪）
    def make_networks():
        act_shape = spec.actions.shape
        policy_net = snt.Sequential([
            snt.Flatten(),
            snt.nets.MLP([256, 256, tf.math.reduce_prod(act_shape)],
                         activation=tf.nn.relu),
            snt.Reshape(act_shape),
            ClipByActSpec(act_low, act_high),  # 手动裁剪
        ])
        critic_net = snt.Sequential([
            snt.Flatten(),
            snt.nets.MLP([256, 256, 1], activation=tf.nn.relu),
        ])
        # 关键：自己包成 DDPGNetworks，不经过 Acme 的 ClipToSpec
        return netlib.DDPGNetworks(
            policy_network=policy_net,
            critic_network=critic_net,
            observation_network=tf_utils.batch_concat,  # 默认展平
        )

    # 3. 构造 agent
    agent = ddpg.DDPG(
        environment_spec=spec,
        networks=make_networks(),  # <-- 用自己做的 networks
        sigma=0.15,
        batch_size=64,
        target_update_period=100,
        logger=loggers.make_default_logger('logs', time_delta=10.),
    )

    # 4. 训练循环
    loop = EnvironmentLoop(env, agent, logger=agent.logger, label='Myo')
    loop.run(num_episodes=500)
    print('训练完成！')
if __name__ == '__main__':
    main()