import dm_env
from dm_env import specs
import gym
import myosuite  # 注册环境

class MyoDmAdapter(dm_env.Environment):
    def __init__(self, gym_env_id: str):
        self.gym = gym.make(gym_env_id)
        self._reset_next = True

    def reset(self):
        obs, _ = self.gym.reset()
        self._reset_next = False
        return dm_env.restart(obs)

    def step(self, action):
        obs, rew, term, trunc, _ = self.gym.step(action)
        done = term or trunc
        if done:
            self._reset_next = True
            return dm_env.termination(reward=rew, observation=obs)
        return dm_env.transition(reward=rew, observation=obs)

    def observation_spec(self):
        s = self.gym.observation_space
        return specs.Array(shape=s.shape, dtype=s.dtype, name="obs")

    def action_spec(self):
        s = self.gym.action_space
        return specs.Array(shape=s.shape, dtype=s.dtype, name="act")