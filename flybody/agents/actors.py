"""Acme agent implementations -- Acme代理实现."""

from typing import Callable #从 Python 标准库 typing 中导入 Callable，用于类型注解，表示一个可调用对象（如函数、类构造函数等）

import numpy as np          #导入 NumPy 库，并将其简写为 np。NumPy 是 Python 中用于科学计算的核心库，主要用于处理数组和矩阵运算

from acme import adders # 从 DeepMind 的强化学习框架 Acme 中导入以下模块：
from acme import core   # core：Acme 的核心接口和抽象类，定义了 Agent 和环境交互的基本结构。
from acme import types  # adders：用于将经验数据（如状态、动作、奖励等）添加到回放缓冲区的工具。

from acme.tf import utils as tf2_utils                      # tf2_utils：TensorFlow 2.x 的工具函数，常用于构建网络、处理变量等。
from acme.tf import variable_utils as tf2_variable_utils    # tf2_variable_utils：用于处理 TensorFlow 变量的工具，比如变量复制、同步等，常用于分布式训练或目标网络更新。

import dm_env           #导入 DeepMind 的 dm_env 库，这是一个用于定义强化学习环境的通用接口，类似于 OpenAI Gym，但更底层、更灵活。
import sonnet as snt    #导入 DeepMind 的神经网络库 Sonnet，并将其简写为 snt。Sonnet 是一个基于 TensorFlow 的神经网络构建库，设计更模块化，适合构建复杂的强化学习模型
import tensorflow as tf #导入 TensorFlow 深度学习框架，并将其简写为 tf。用于构建和训练神经网络模型。
import tensorflow_probability as tfp
#导入 TensorFlow Probability（TFP），这是 TensorFlow 的概率建模库，支持概率分布、贝叶斯神经网络、随机变量等，常用于策略梯度、不确定性建模等强化学习算法中。

tfd = tfp.distributions


class DelayedFeedForwardActor(core.Actor):
    """
    A feed-forward actor with optional delay between observation and action.
    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    一种具有可选观察与行动之间延迟的前馈型执行器。
    基于前馈策略的执行器，它接收非批量化的观察结果，并输出非批量化的行动。
    它还允许将经验添加到回放中，并从学习器更新策略的权重。
    """

    def __init__(
        self,
        policy_network: snt.Module,
        #= None：默认值就是 None，调用时可以不写这个参数，代码也不会报错
        adder: adders.Adder | None = None,
        variable_client: tf2_variable_utils.VariableClient | None = None,
        action_delay: int | None = None,
        observation_callback: Callable | None = None,
    ):
        """Initializes the actor.
        Args:
            Policy_network：要运行的策略。
            adder：允许向a添加经验的加法器对象数据集/回放缓冲区。
            Variable_client：对象，允许从学习者副本中复制权重策略到参与者副本（如果它们是分开的）。
            Action_delay：延迟动作的时间步数。
            observation_callback：可选调用，用于处理之前的观察将它们传递给政策制定者
            policy_network: the policy to run.
            adder: the adder object to which allows to add experiences to a
                dataset/replay buffer.
            variable_client: object which allows to copy weights from the learner copy
                of the policy to the actor copy (in case they are separate).
            action_delay: number of timesteps to delay the action for.
            observation_callback: Optional callable to process observations before
                passing them to policy.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._policy_network = policy_network
        self._action_delay = action_delay
        if action_delay is not None:
            self._action_queue = []
        self._observation_callback = observation_callback

    '''
    @tf.function是 TensorFlow 的“加速器”，它把你的 Python 函数变成高效的计算图，
    适合训练、推理和部署，但要注意它不是“万能胶”，写的时候要符合图模式的规范。
    observation: types.NestedArray是参数 observation 的类型注解，说明它必须是 Acme 定义的“嵌套数组”结构（可以是 np.ndarray、嵌套 dict/list 等）。
    -> types.NestedTensor：函数返回值的类型注解，说明该函数会返回一个“嵌套的 TensorFlow 张量”结构（可以是 tf.Tensor、嵌套 dict/list 等）。
    观察-->策略-->动作
    '''
    @tf.function
    def _policy(self, observation: types.NestedArray) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        #添加一个虚拟批处理尺寸，并将numpy转换为TF。
        batched_observation = tf2_utils.add_batch_dim(observation)  

        # Compute the policy, conditioned on the observation.
        #根据批量观察结果计算策略
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        #从随机策略中抽样
        #对样本进行采样，如果策略是TensorFlow Probability 的分布对象，按该分布采样一个随机动作，否则直接返回策略
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy
        return action

    def select_action(self,
                      observation: types.NestedArray) -> np.ndarray:
        """Samples from the policy and returns an action. -- 从策略中采样并返回一个动作。"""
        #如果回调观察不为空，则使用回调函数处理观察结果
        if self._observation_callback is not None:
            observation = self._observation_callback(observation)
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Maybe delay action -- 可能存在延迟行动.
        if self._action_delay is not None:
            # 若行动队列的长度小于行动延迟，则将当前行动添加到队列中，并返回一个全零的行动（表示在填充初始队列时不采取任何行动）。
            # 否则，将当前行动添加到队列中，并弹出并返回队列中的第一个行动（表示采取最早的行动）。
            # 也就是你在排队打饭的时候如果没有队伍那就可以直接打饭，如果有就加入队伍并需要时间来最终排到你打饭
            if len(self._action_queue) < self._action_delay:
                self._action_queue.append(action)
                action = 0 * action  # Return 0 while filling the initial queue.
            else:
                self._action_queue.append(action)
                action = self._action_queue.pop(0)

        # Return a numpy array with squeezed out batch dimension -- 返回一个压缩批量尺寸的numpy数组.
        return tf2_utils.to_numpy_squeeze(action)


    '''
    #不是很理解
    重置环境  →  observe_first(初始状态)
    ↓
    选动作 → 执行 → 得到 next_timestep
    ↓
    observe(动作, next_timestep)
    ↓
    （循环 N 步）
    ↓
    回合结束 → 训练端更新参数 →  update() 拉新权重
    '''
    # 环境刚重置、还没产生任何动作时，先调用这里。
    # 只把初始状态（带 FIRST 标签的 TimeStep）写进回放池，作为一条轨迹的“开头”。
    # 如果外部没给 adder（None），就什么都不做，避免报错。
    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    # 在上一步动作 action 已经执行后，环境返回了 next_timestep（包含奖励、新状态、是否结束）。
    # 把“动作 + 后续时间步”这对组合写进回放池，用来后面做经验回放或 GAE 计算。
    # 同样，没有 adder 就跳过，代码保持健壮。
    def observe(self, action: types.NestedArray,
                next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    # 如果构造时给了 variable_client（通常指向参数服务器或目标网络），就拉一次最新权重。
    # wait=True 表示阻塞直到拉完；False 表示异步后台拉，继续干别的。
    # 没有客户端就什么都不做，单机单进程场景直接忽略。
    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)
