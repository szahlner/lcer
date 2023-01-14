import gym
import numpy as np
from gym import spaces


class AntTruncatedV2Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_obs_keep = 27
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n_obs_keep,))
        self._max_episode_steps = env._max_episode_steps

    def observation(self, obs):
        return obs[: self.n_obs_keep]
