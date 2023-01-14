import os

import numpy as np
import pytest

from lcer.common.replay_memory import BaseReplayMemory


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim",
    [
        (int(1e6), 123, 10, 3),
        (int(1e3), 123, 100, 30),
    ],
)
def test_base_replay_buffer_initialization_and_attributes(capacity, seed, state_dim, action_dim):
    memory = BaseReplayMemory(
        capacity=capacity,
        seed=seed,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # capacity
    assert hasattr(memory, "capacity")
    assert isinstance(memory.capacity, int)

    # size
    assert hasattr(memory, "size")
    assert isinstance(memory.size, int)

    # position
    assert hasattr(memory, "position")
    assert isinstance(memory.position, int)

    # action_dim
    assert hasattr(memory, "action_dim")
    assert isinstance(memory.action_dim, int)

    # state_dim
    assert hasattr(memory, "state_dim")
    assert isinstance(memory.state_dim, int)

    # buffer
    assert hasattr(memory, "buffer")
    assert isinstance(memory.buffer, dict)
    assert "state" in memory.buffer
    assert "next_state" in memory.buffer
    assert "action" in memory.buffer
    assert "reward" in memory.buffer
    assert "mask" in memory.buffer


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim,n_test",
    [
        (int(1e6), 123, 10, 3, 10),
        (int(1e3), 123, 100, 30, 100),
    ],
)
def test_base_replay_buffer_push(capacity, seed, state_dim, action_dim, n_test):
    memory = BaseReplayMemory(
        capacity=capacity,
        seed=seed,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    state = np.empty(shape=(1, state_dim))
    next_state = np.empty(shape=(1, state_dim))
    action = np.empty(shape=(1, action_dim))
    reward = 0.0
    mask = True

    for _ in range(n_test):
        memory.push(state, action, reward, next_state, mask)

    assert memory.size == n_test
    assert memory.position == n_test
    assert len(memory) == n_test


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim,n_test",
    [
        (int(1e6), 123, 10, 3, 10),
        (int(1e3), 123, 100, 30, 100),
    ],
)
def test_base_replay_buffer_sample(capacity, seed, state_dim, action_dim, n_test):
    memory = BaseReplayMemory(
        capacity=capacity,
        seed=seed,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    state = np.empty(shape=(1, state_dim))
    next_state = np.empty(shape=(1, state_dim))
    action = np.empty(shape=(1, action_dim))
    reward = 0.0
    mask = True

    for _ in range(n_test):
        memory.push(state, action, reward, next_state, mask)

    state, action, reward, next_state, mask = memory.sample(batch_size=n_test)

    assert state.shape == (n_test, state_dim)
    assert action.shape == (n_test, action_dim)
    assert reward.shape == (n_test, 1)
    assert next_state.shape == (n_test, state_dim)
    assert mask.shape == (n_test, 1)

    n_test = n_test // 2
    state, action, reward, next_state, mask = memory.sample(batch_size=n_test)

    assert state.shape == (n_test, state_dim)
    assert action.shape == (n_test, action_dim)
    assert reward.shape == (n_test, 1)
    assert next_state.shape == (n_test, state_dim)
    assert mask.shape == (n_test, 1)


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim",
    [
        (int(1e6), 123, 10, 3),
        (int(1e3), 123, 100, 30),
    ],
)
def test_base_replay_buffer_capacity_position_and_size(capacity, seed, state_dim, action_dim):
    memory = BaseReplayMemory(
        capacity=capacity,
        seed=seed,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    state = np.empty(shape=(1, state_dim))
    next_state = np.empty(shape=(1, state_dim))
    action = np.empty(shape=(1, action_dim))
    reward = 0.0
    mask = True

    for _ in range(capacity):
        memory.push(state, action, reward, next_state, mask)

    assert memory.size == memory.capacity
    assert memory.position == 0

    memory.push(state, action, reward, next_state, mask)

    assert memory.size == memory.capacity
    assert memory.position == 1


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim",
    [
        (10, 123, 10, 3),
        (100, 123, 100, 30),
    ],
)
def test_base_replay_buffer_save_and_load(capacity, seed, state_dim, action_dim):
    env_name = "env_name"
    suffix = "suffix_base_replay_buffer_{}".format(capacity)
    directory = "checkpoints"
    save_path = os.path.join(directory, "sac_buffer_{}_{}.pkl".format(env_name, suffix))

    memory = BaseReplayMemory(
        capacity=capacity,
        seed=seed,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    state = np.empty(shape=(1, state_dim))
    next_state = np.empty(shape=(1, state_dim))
    action = np.empty(shape=(1, action_dim))
    reward = 0.0
    mask = True

    for _ in range(capacity):
        memory.push(state, action, reward, next_state, mask)

    memory.save_buffer(env_name=env_name, suffix=suffix, save_path=None)

    assert os.path.exists(directory)
    assert os.path.isdir(directory)
    assert os.path.exists(save_path)

    # Delete memory
    del memory

    memory = BaseReplayMemory(
        capacity=capacity,
        seed=seed,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    memory.load_buffer(save_path=save_path)

    assert memory.size == memory.capacity
    assert memory.position == 0

    # Clean up
    os.remove(save_path)
    if len(os.listdir(directory)) == 0:
        os.rmdir(directory)

        assert not os.path.exists(directory)
        assert not os.path.isdir(directory)
    else:
        assert os.path.exists(directory)
        assert os.path.isdir(directory)

    # Wrong load
    with pytest.raises(FileNotFoundError):
        memory.load_buffer(save_path=save_path)
