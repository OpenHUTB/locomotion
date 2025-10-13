"""Distributional MPO learner implementation. -- 分布式MPO学习器实现"""

import time                             #用于测量时间、控制节奏、记录时间戳等
from typing import List, Optional       #提供类型注解，如列表、可选类型，增强代码可读性
import re                               #正则表达式模块，用于字符串匹配等（可能备用）

import acme                             #强化学习库，提供构建和训练RL代理的工具
from acme import types                  #定义了一些常用的类型别名和数据结构，方便在代码中使用和理解。
from acme.tf import losses              #提供强化学习中常用的损失函数（如策略梯度损失等）
from acme.tf import networks            #提供预定义的神经网络结构（策略网、Q网等）
from acme.tf import savers as tf2_savers#提供 TensorFlow 模型保存与加载工具
from acme.tf import utils as tf2_utils  #TensorFlow 2.x 的工具函数，常用于构建网络、处理变量等。
from acme.utils import counting         #用于跟踪和计数训练过程中的各种指标（如步骤数、时间等）
from acme.utils import loggers          #用于日志记录和监控训练过程（如写入 TensorBoard 等）
import numpy as np                      #科学计算库，用于数组、数值运算
import sonnet as snt                    #DeepMind 的神经网络库，用于构建模块化模型
import tensorflow as tf                 #Google 的深度学习框架，用于训练和推理


class DistributionalMPOLearner(acme.Learner):
    """Distributional MPO learner. -- 分布式最大后验策略优化学习器"""

    def __init__(
        self,
        #snt.Module是一种特殊的类，当你定义神经网络时继承它，就能拥有自动变量管理等功能
        policy_network: snt.Module,
        critic_network: snt.Module,
        target_policy_network: snt.Module,
        target_critic_network: snt.Module,
        discount: float,
        num_samples: int,
        target_policy_update_period: int,
        target_critic_update_period: int,
        dataset: tf.data.Dataset,
        observation_network: types.TensorTransformation = tf.identity,
        target_observation_network: types.TensorTransformation = tf.identity,
        policy_loss_module: Optional[snt.Module] = None,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        dual_optimizer: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint_enable: bool = True,
        checkpoint_max_to_keep: Optional[
            int] = 1,  # If None, all checkpoints are kept.
        directory: str | None = '~/acme/',
        checkpoint_to_load: Optional[str] = None,
        time_delta_minutes: float = 30.,
    ):

        # Store online and target networks -- 在线存储和目标网络.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network

        # Make sure observation networks are snt.Module's so they have variables -- 
        # 请确保你定义的 observation networks（观察/预处理网络）是继承自 snt.Module（Sonnet 模块）的类，
        # 这样 Sonnet 才能自动管理它们内部的神经网络参数（即 tf.Variables），从而使得这些参数能够正常参与训练、保存和加载等操作​.
        self._observation_network = tf2_utils.to_sonnet_module(
            observation_network)
        self._target_observation_network = tf2_utils.to_sonnet_module(
            target_observation_network)

        # General learner book-keeping and loggers -- 一般学习者簿记和记录员.
        self._counter = counter or counting.Counter()   #如果没有提供计数器(counter = None)，就创建一个新的计数器
        self._logger = logger or loggers.make_default_logger('learner')

        # Other learner parameters  -- 其他学习器参数.
        self._discount = discount
        self._num_samples = num_samples
        self._clipping = clipping

        # Necessary to track when to update target networks -- 当更新目标网络追踪是必要的.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_policy_update_period = target_policy_update_period
        self._target_critic_update_period = target_critic_update_period

        # Batch dataset and create iterator -- 批处理数据集并创造迭代器.
        '''
        # TODO(b/155086959): Fix type stubs and remove -- 是一个典型的 ​​TODO 注释​​，
        # 通常出现在代码库（尤其是大型工程如 Google 内部项目、TensorFlow、Sonnet、Acme 等）中，用来标记 ​​待完成的改进任务​​，
        # 并且通常关联了一个 ​​问题追踪编号（Issue / Bug ID）​​，这里是：b/155086959。.
        '''
        self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types


        '''
        是一个非常典型的 Python ​​条件赋值（短路逻辑赋值）​​，用于初始化一个成员变量 self._policy_loss_module，
        它的作用通常是设置一个用于计算​​策略损失（policy loss）​​的模块，在这里使用的是来自 losses模块的 
        ​​MPO（Maximum a Posteriori Policy Optimization，最大后验策略优化）算法的损失模块​​。
        '''
        self._policy_loss_module = policy_loss_module or losses.MPO(
            epsilon=1e-1,
            epsilon_penalty=1e-3,
            epsilon_mean=1e-3,
            epsilon_stddev=1e-6,
            init_log_temperature=1.,
            init_log_alpha_mean=1.,
            init_log_alpha_stddev=10.)

        # Create the optimizers.
        # snt.optimizers.Adam(1e-4)：创建一个 Sonnet 封装的 Adam 优化器，其学习率（learning rate）为 0.0001 (即 1×10⁻⁴)。
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)  
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
        self._dual_optimizer = dual_optimizer or snt.optimizers.Adam(1e-2)

        # Expose the variables.
        '''
        使用了 Sonnet 的 snt.Sequential，这是一个​​容器模块​​，它将多个 Sonnet 模块按顺序组合在一起，前一个模块的输出作为后一个模块的输入。
        这里将两个网络模块按顺序组合：
        self._target_observation_network：目标观察网络，可能用于对原始输入（如状态/观测）进行特征提取。
        self._target_policy_network：目标策略网络，根据提取的特征决定采取的动作。
        👉 所以，policy_network_to_expose是一个 ​​组合策略网络​​，输入原始观测，
        先经过观察网络处理，再送到策略网络输出动作。它本身也是一个 snt.Module，并且具有自己的可训练参数（variables）。
        '''
        policy_network_to_expose = snt.Sequential(
            [self._target_observation_network, self._target_policy_network])
        # 以字典形式存储目标评论家网络和策略网络的变量，方便外部访问和管理
        self._variables = {
            'critic': self._target_critic_network.variables,
            'policy': policy_network_to_expose.variables,
        }

        # Create a checkpointer and snapshotter object  --  创建一个检查指针和快照对象.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint_enable:
            self._checkpointer = tf2_savers.Checkpointer(
                subdirectory='dmpo_learner',
                objects_to_save={
                    'counter': self._counter,
                    'policy': self._policy_network,
                    'critic': self._critic_network,
                    'observation': self._observation_network,
                    'target_policy': self._target_policy_network,
                    'target_critic': self._target_critic_network,
                    'target_observation': self._target_observation_network,
                    'policy_optimizer': self._policy_optimizer,
                    'critic_optimizer': self._critic_optimizer,
                    'dual_optimizer': self._dual_optimizer,
                    'policy_loss_module': self._policy_loss_module,
                    'num_steps': self._num_steps
                },
                directory=directory,
                time_delta_minutes=time_delta_minutes,
                max_to_keep=checkpoint_max_to_keep,
            )

            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={
                    'policy-0':
                    snt.Sequential([
                        self._target_observation_network,
                        self._target_policy_network
                    ])
                },
                directory=directory,
                time_delta_minutes=time_delta_minutes)

        # Maybe load checkpoint.
        if checkpoint_to_load is not None:
            _checkpoint = tf.train.Checkpoint(
                counter=tf2_savers.SaveableAdapter(self._counter),
                policy=self._policy_network,
                critic=self._critic_network,
                observation=self._observation_network,
                target_policy=self._target_policy_network,
                target_critic=self._target_critic_network,
                target_observation=self._target_observation_network,
                policy_optimizer=self._policy_optimizer,
                critic_optimizer=self._critic_optimizer,
                dual_optimizer=self._dual_optimizer,
                policy_loss_module=self._policy_loss_module,
                num_steps=self._num_steps,
            )
            status = _checkpoint.restore(checkpoint_to_load)  # noqa: F841
            # The assert below will not work because at this point not all variables have
            # been created in tf.train.Checkpoint argument objects. However, it's good
            # enough to revive a job from its checkpoint. Another option is to put
            # the checkpoint loading in the self.step method below, then the assertion
            # will work.
            # status.assert_consumed()  # Sanity check.

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _step(self) -> types.NestedTensor:
        # Update target network.
        online_policy_variables = self._policy_network.variables
        target_policy_variables = self._target_policy_network.variables
        online_critic_variables = (
            *self._observation_network.variables,
            *self._critic_network.variables,
        )
        target_critic_variables = (
            *self._target_observation_network.variables,
            *self._target_critic_network.variables,
        )

        # Make online policy -> target policy network update ops.
        if tf.math.mod(self._num_steps,
                       self._target_policy_update_period) == 0:
            for src, dest in zip(online_policy_variables,
                                 target_policy_variables):
                dest.assign(src)
        # Make online critic -> target critic network update ops.
        if tf.math.mod(self._num_steps,
                       self._target_critic_update_period) == 0:
            for src, dest in zip(online_critic_variables,
                                 target_critic_variables):
                dest.assign(src)

        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)
        transitions: types.Transition = inputs.data

        # Get batch size and scalar dtype.
        batch_size = transitions.reward.shape[0]

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

        with tf.GradientTape(persistent=True) as tape:
            # Maybe transform the observation before feeding into policy and critic.
            # Transforming the observations this way at the start of the learning
            # step effectively means that the policy and critic share observation
            # network weights.
            o_tm1 = self._observation_network(transitions.observation)
            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t = tf.stop_gradient(
                self._target_observation_network(transitions.next_observation))

            # Get online and target action distributions from policy networks.
            online_action_distribution = self._policy_network(o_t)
            target_action_distribution = self._target_policy_network(o_t)

            # Sample actions to evaluate policy; of size [N, B, ...].
            sampled_actions = target_action_distribution.sample(
                self._num_samples)

            # Tile embedded observations to feed into the target critic network.
            # Note: this is more efficient than tiling before the embedding layer.
            tiled_o_t = tf2_utils.tile_tensor(o_t,
                                              self._num_samples)  # [N, B, ...]

            # Compute target-estimated distributional value of sampled actions at o_t.
            sampled_q_t_distributions = self._target_critic_network(
                # Merge batch dimensions; to shape [N*B, ...].
                snt.merge_leading_dims(tiled_o_t, num_dims=2),
                snt.merge_leading_dims(sampled_actions, num_dims=2))

            # Compute average logits by first reshaping them and normalizing them
            # across atoms.
            new_shape = [self._num_samples, batch_size, -1]  # [N, B, A]
            sampled_logits = tf.reshape(sampled_q_t_distributions.logits,
                                        new_shape)
            sampled_logprobs = tf.math.log_softmax(sampled_logits, axis=-1)
            averaged_logits = tf.reduce_logsumexp(sampled_logprobs, axis=0)

            # Construct the expected distributional value for bootstrapping.
            q_t_distribution = networks.DiscreteValuedDistribution(
                values=sampled_q_t_distributions.values,
                logits=averaged_logits)

            # Compute online critic value distribution of a_tm1 in state o_tm1.
            q_tm1_distribution = self._critic_network(o_tm1,
                                                      transitions.action)

            # Compute critic distributional loss.
            critic_loss = losses.categorical(q_tm1_distribution,
                                             transitions.reward,
                                             discount * transitions.discount,
                                             q_t_distribution)
            critic_loss = tf.reduce_mean(critic_loss)

            # Compute Q-values of sampled actions and reshape to [N, B].
            sampled_q_values = sampled_q_t_distributions.mean()
            sampled_q_values = tf.reshape(sampled_q_values,
                                          (self._num_samples, -1))

            # Compute MPO policy loss.
            policy_loss, policy_stats = self._policy_loss_module(
                online_action_distribution=online_action_distribution,
                target_action_distribution=target_action_distribution,
                actions=sampled_actions,
                q_values=sampled_q_values)

        # For clarity, explicitly define which variables are trained by which loss.
        critic_trainable_variables = (
            # In this agent, the critic loss trains the observation network.
            self._observation_network.trainable_variables +
            self._critic_network.trainable_variables)
        policy_trainable_variables = self._policy_network.trainable_variables
        # The following are the MPO dual variables, stored in the loss module.
        dual_trainable_variables = self._policy_loss_module.trainable_variables

        # Compute gradients.
        critic_gradients = tape.gradient(critic_loss,
                                         critic_trainable_variables)
        policy_gradients, dual_gradients = tape.gradient(
            policy_loss,
            (policy_trainable_variables, dual_trainable_variables))

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients = tuple(
                tf.clip_by_global_norm(policy_gradients, 40.)[0])
            critic_gradients = tuple(
                tf.clip_by_global_norm(critic_gradients, 40.)[0])

        # Apply gradients.
        self._critic_optimizer.apply(critic_gradients,
                                     critic_trainable_variables)
        self._policy_optimizer.apply(policy_gradients,
                                     policy_trainable_variables)
        self._dual_optimizer.apply(dual_gradients, dual_trainable_variables)

        # Losses to track.
        fetches = {
            'critic_loss': critic_loss,
            'policy_loss': policy_loss,
        }
        fetches.update(policy_stats)  # Log MPO stats.

        return fetches

    def step(self):
        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
            self._checkpointer.save()

        if self._snapshotter is not None:
            if self._snapshotter.save():
                # Log at what actor_steps this snapshot was saved. The actor_steps
                # counter is updated at end of episode so fetches['actor_steps']
                # may not exist yet when the first policy snapshot is saved.
                actor_steps = fetches['actor_steps'] if 'actor_steps' in fetches else 0
                fetches['saved_snapshot_at_actor_steps'] = actor_steps
                # Increment the snapshot counter (directly in the snapshotter's path).
                for path in list(self._snapshotter._snapshots.keys()):
                    snapshot = self._snapshotter._snapshots[path]  # noqa: F841
                    # Assume that path ends with, e.g., "/policy-17".
                    # Find sequence of digits at end of string.
                    current_counter = re.findall('[\d]+$', path)[0]
                    new_path = path.replace(
                        'policy-' + current_counter,
                        'policy-' + str(int(current_counter) + 1))
                    # Redirect snapshot to new key and delete old key.
                    self._snapshotter._snapshots[
                        new_path] = self._snapshotter._snapshots.pop(path)
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        self._counter.increment(get_variables_calls=1)
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]
