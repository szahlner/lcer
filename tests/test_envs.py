import gym
import pytest

from lcer.common.env_checker import check_env
from lcer.common.envs.wrapper import AntTruncatedV2Wrapper


@pytest.mark.parametrize("env_id", ["Ant-v2", "Hopper-v2"])
def test_custom_envs(env_id):
    env = gym.make(env_id)

    # Check for Ant-v2 and apply AntTruncatedV2Wrapper
    if env_id == "Ant-v2":
        env = AntTruncatedV2Wrapper(env)

    check_env(env)


@pytest.mark.parametrize("env_id", ["Ant-v2"])
def test_ant_truncated_v2_observation_dimensions(env_id):
    env = gym.make(env_id)

    # Check for Ant-v2 and apply AntTruncatedV2Wrapper
    if env_id == "Ant-v2":
        env = AntTruncatedV2Wrapper(env)

    assert env.observation_space.shape == (27,)
