import os

import gym
import numpy as np
import pytest
import shadowhand_gym

from lcer.common.her.replay_memory import HerReplayMemory, HerSampler, Normalizer
from lcer.common.utils import get_env_params


@pytest.mark.parametrize(
    "env_id,replay_strategy,replay_k,buffer_size,normalize",
    [
        ("ShadowHandReach-v1", "future", 4, int(1e6), True),
        ("ShadowHandReach-v1", "future", 4, int(1e3), False),
    ],
)
def test_her_replay_buffer_initialization_and_attributes(env_id, replay_strategy, replay_k, buffer_size, normalize):
    env = gym.make(env_id)
    env_params = get_env_params(env)
    sampler = HerSampler(replay_strategy=replay_strategy, replay_k=replay_k, reward_func=env.compute_reward)

    memory = HerReplayMemory(
        env_params=env_params, buffer_size=buffer_size, sample_func=sampler.sample_her_transitions, normalize=normalize
    )

    # env_params
    assert hasattr(memory, "env_params")
    assert isinstance(memory.env_params, dict)

    # T
    assert hasattr(memory, "T")
    assert isinstance(memory.T, int)

    # size
    assert hasattr(memory, "size")
    assert isinstance(memory.size, int)

    # normalize
    assert hasattr(memory, "normalize")
    assert isinstance(memory.normalize, bool)

    # current_size
    assert hasattr(memory, "current_size")
    assert isinstance(memory.current_size, int)

    # n_transitions_stored
    assert hasattr(memory, "n_transitions_stored")
    assert isinstance(memory.n_transitions_stored, int)

    # sample_func
    assert hasattr(memory, "sample_func")

    # buffers
    assert hasattr(memory, "buffers")
    assert isinstance(memory.buffers, dict)

    if normalize:
        # o_norm
        assert hasattr(memory, "o_norm")
        assert isinstance(memory.o_norm, Normalizer)

        # g_norm
        assert hasattr(memory, "g_norm")
        assert isinstance(memory.g_norm, Normalizer)


@pytest.mark.parametrize(
    "env_id,replay_strategy,replay_k,buffer_size,normalize",
    [
        ("ShadowHandReach-v1", "future", 4, int(1e6), True),
        ("ShadowHandReach-v1", "future", 4, int(1e3), False),
    ],
)
def test_her_replay_buffer_push(env_id, replay_strategy, replay_k, buffer_size, normalize):
    env = gym.make(env_id)
    env_params = get_env_params(env)
    sampler = HerSampler(replay_strategy=replay_strategy, replay_k=replay_k, reward_func=env.compute_reward)

    memory = HerReplayMemory(
        env_params=env_params, buffer_size=buffer_size, sample_func=sampler.sample_her_transitions, normalize=normalize
    )

    state = env.reset()
    action = env.action_space.sample()

    ep_state, ep_ag, ep_g, ep_actions = [], [], [], []
    for _ in range(env_params["max_timesteps"]):
        ep_state.append(state["observation"])
        ep_ag.append(state["achieved_goal"])
        ep_g.append(state["desired_goal"])
        ep_actions.append(action)
    ep_state.append(state["observation"])
    ep_ag.append(state["achieved_goal"])

    ep_state = np.array([ep_state])
    ep_ag = np.array([ep_ag])
    ep_g = np.array([ep_g])
    ep_actions = np.array([ep_actions])

    memory.push_episode([ep_state, ep_ag, ep_g, ep_actions])

    assert memory.current_size == 1  # Pushed 1 episode
    assert memory.n_transitions_stored == env_params["max_timesteps"]  # 1 episode = max_timesteps
