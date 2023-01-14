import os

import numpy as np
import pytest

from lcer.common.replay_memory import ReplayMemory


@pytest.mark.parametrize(
    "capacity,seed",
    [
        (int(1e6), 123),
        (int(1e3), 123),
    ],
)
def test_replay_buffer_initialization_and_attributes(capacity, seed):
    memory = ReplayMemory(capacity=capacity, seed=seed)

    # capacity
    assert hasattr(memory, "capacity")
    assert isinstance(memory.capacity, int)

    # position
    assert hasattr(memory, "position")
    assert isinstance(memory.position, int)

    # buffer
    assert hasattr(memory, "buffer")
    assert isinstance(memory.buffer, list)


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim,n_test",
    [
        (int(1e6), 123, 10, 3, 10),
        (int(1e3), 123, 100, 30, 100),
    ],
)
def test_replay_buffer_push(capacity, seed, state_dim, action_dim, n_test):
    memory = ReplayMemory(capacity=capacity, seed=seed)

    state = np.empty(shape=(state_dim,))
    next_state = np.empty(shape=(state_dim,))
    action = np.empty(shape=(action_dim,))
    reward = 0.0
    mask = True

    for _ in range(n_test):
        memory.push(state, action, reward, next_state, mask)

    assert memory.position == n_test
    assert len(memory) == n_test


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim,n_test",
    [
        (int(1e6), 123, 10, 3, 10),
        (int(1e3), 123, 100, 30, 100),
    ],
)
def test_replay_buffer_sample(capacity, seed, state_dim, action_dim, n_test):
    memory = ReplayMemory(capacity=capacity, seed=seed)

    state = np.empty(shape=(state_dim,))
    next_state = np.empty(shape=(state_dim,))
    action = np.empty(shape=(action_dim,))
    reward = 0.0
    mask = True

    for _ in range(n_test):
        memory.push(state, action, reward, next_state, mask)

    state, action, reward, next_state, mask = memory.sample(batch_size=n_test)

    assert state.shape == (n_test, state_dim)
    assert action.shape == (n_test, action_dim)
    assert reward.shape == (n_test,)
    assert next_state.shape == (n_test, state_dim)
    assert mask.shape == (n_test,)

    n_test = n_test // 2
    state, action, reward, next_state, mask = memory.sample(batch_size=n_test)

    assert state.shape == (n_test, state_dim)
    assert action.shape == (n_test, action_dim)
    assert reward.shape == (n_test,)
    assert next_state.shape == (n_test, state_dim)
    assert mask.shape == (n_test,)


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim",
    [
        (int(1e6), 123, 10, 3),
        (int(1e3), 123, 100, 30),
    ],
)
def test_replay_buffer_capacity_position_and_size(capacity, seed, state_dim, action_dim):
    memory = ReplayMemory(capacity=capacity, seed=seed)

    state = np.empty(shape=(state_dim,))
    next_state = np.empty(shape=(state_dim,))
    action = np.empty(shape=(action_dim,))
    reward = 0.0
    mask = True

    for _ in range(capacity):
        memory.push(state, action, reward, next_state, mask)

    assert len(memory) == memory.capacity
    assert memory.position == 0

    memory.push(state, action, reward, next_state, mask)

    assert len(memory) == memory.capacity
    assert memory.position == 1


@pytest.mark.parametrize(
    "capacity,seed,state_dim,action_dim",
    [
        (10, 123, 10, 3),
        (100, 123, 100, 30),
    ],
)
def test_replay_buffer_save_and_load(capacity, seed, state_dim, action_dim):
    env_name = "env_name"
    suffix = "suffix_replay_buffer_{}".format(capacity)
    directory = "checkpoints"
    save_path = os.path.join(directory, "sac_buffer_{}_{}.pkl".format(env_name, suffix))

    memory = ReplayMemory(capacity=capacity, seed=seed)

    state = np.empty(shape=(state_dim,))
    next_state = np.empty(shape=(state_dim,))
    action = np.empty(shape=(action_dim,))
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

    memory = ReplayMemory(capacity=capacity, seed=seed)
    memory.load_buffer(save_path=save_path)

    assert len(memory) == memory.capacity
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
