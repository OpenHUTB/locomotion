import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.*")
import dm_env
from dm_env import specs
import gymnasium as gym
import myosuite  # 注册环境


class MyoDmAdapter(dm_env.Environment):
    def __init__(self, gym_env_id: str):
        """
        gym_env_id：字符串，指定 MyoSuite 中注册的环境 ID
        self.gym：保存 Gymnasium 环境实例。
        self._reset_next：标志位，指示下一步是否需要重置环境（初始为 True）。
        """
        self.gym = gym.make(gym_env_id)
        self._reset_next = True

    def reset(self):
        """重置 Gymnasium 环境，并返回 dm_env 格式的初始时间步（restart）"""
        obs, _ = self.gym.reset()
        self._reset_next = False
        return dm_env.restart(obs)

    def step(self, action):
        """
        作用：执行一步动作，返回 dm_env 格式的时间步。
        参数：action：动作向量，符合 Gymnasium 环境的动作空间。
        """
        obs, rew, term, trunc, _ = self.gym.step(action)
        done = term or trunc
        if done:
            self._reset_next = True
            return dm_env.termination(reward=rew, observation=obs)
        return dm_env.transition(reward=rew, observation=obs)

    def observation_spec(self):
        """
        作用：定义观测空间的规范（shape、dtype、name）。
        返回值：dm_env.specs.Array，描述观测空间的结构。
        用途：供 RL 框架（如 Acme）构建神经网络输入层（观察输入动作输出）。
        """
        s = self.gym.observation_space
        return specs.Array(shape=s.shape, dtype=s.dtype, name="obs")

    def action_spec(self):
        """
        作用：定义动作空间的规范。
        返回值：dm_env.specs.Array，描述动作空间的结构。
        用途：供 RL 框架构建策略网络输出层（观察输入动作输出）。
        """
        s = self.gym.action_space
        return specs.Array(shape=s.shape, dtype=s.dtype, name="act")

    def reward_spec(self):
        """
        作用：定义奖励的规范。
        用途：告诉 RL 框架奖励是标量浮点数。
        """
        return specs.Array(shape=(), dtype="float32", name="reward")

if __name__ == "__main__":
    env = MyoDmAdapter("myoElbowPose1D6MRandom-v0")

    # 打印观测与动作空间
    print(env.observation_spec())
    print(env.action_spec())

    # 测试循环
    ts = env.reset()
    print("init obs:", ts.observation)
    for t in range(100):
        """
        循环执行 100 步随机动作（generate_value() 生成合法随机动作）。
        如果 episode 提前结束（ts.last()），打印结束步数并退出。
        """
        ts = env.step(env.action_spec().generate_value())
        if ts.last():
            print("done at step", t)
            break
    print("test finished")