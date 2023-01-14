import gym
import numpy as np
import torch


def create_log_gaussian(mean: torch.Tensor, log_std: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * np.log(2 * np.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def log_sum_exp(inputs: torch.Tensor, dim: int = None, keep_dim: bool = False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keep_dim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def termination_fn(env_name: str, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray):
    if env_name == "Hopper-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (
            np.isfinite(next_obs).all(axis=-1)
            * np.abs(next_obs[:, 1:] < 100).all(axis=-1)
            * (height > 0.7)
            * (np.abs(angle) < 0.2)
        )

        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "Walker2d-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "Ant-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        x = next_obs[:, 0]
        not_done = np.isfinite(next_obs).all(axis=-1) * (x >= 0.2) * (x <= 1.0)

        done = ~not_done
        done = done[:, None]
        return done
    elif env_name == "InvertedPendulum-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        notdone = np.isfinite(next_obs).all(axis=-1) * (np.abs(next_obs[:, 1]) <= 0.2)
        done = ~notdone
        done = done[:, None]
        return done
    elif env_name == "InvertedDoublePendulum-v2":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        sin1, cos1 = next_obs[:, 1], next_obs[:, 3]
        sin2, cos2 = next_obs[:, 2], next_obs[:, 4]
        theta_1 = np.arctan2(sin1, cos1)
        theta_2 = np.arctan2(sin2, cos2)
        y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

        done = y <= 1
        done = done[:, None]
        return done
    else:
        # HalfCheetah-v2 goes in here too
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        return np.zeros((len(obs), 1), dtype=np.bool)


def get_predicted_states(model, state, action, env_name, deterministic=False):
    inputs = np.concatenate((state, action), axis=-1)

    ensemble_model_means, ensemble_model_vars = model.predict(inputs)
    ensemble_model_means[:, :, 1:] += state

    if deterministic:
        ensemble_samples = ensemble_model_means
    else:
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

    num_models, batch_size, _ = ensemble_model_means.shape
    model_idxes = np.random.choice(model.elite_model_idxes, size=batch_size)
    batch_idxes = np.arange(0, batch_size)

    samples = ensemble_samples[model_idxes, batch_idxes]
    new_reward, new_next_state = samples[:, :1], samples[:, 1:]

    new_done = termination_fn(env_name, state, action, new_next_state)

    return new_reward, new_next_state, new_done


def get_predicted_states_her(model, state, ag, g, action, env_params, deterministic=False):
    inputs = np.concatenate((state, ag, g, action), axis=-1)

    ensemble_model_means, ensemble_model_vars = model.predict(inputs)
    ensemble_model_means += np.concatenate((state, ag, g), axis=-1)

    if deterministic:
        ensemble_samples = ensemble_model_means
    else:
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

    num_models, batch_size, _ = ensemble_model_means.shape
    model_idxes = np.random.choice(model.elite_model_idxes, size=batch_size)
    batch_idxes = np.arange(0, batch_size)

    samples = ensemble_samples[model_idxes, batch_idxes]
    new_next_state, new_next_state_ag = (
        samples[:, : env_params["obs"]],
        samples[:, env_params["obs"] : env_params["obs"] + env_params["goal"]],
    )

    return new_next_state, new_next_state_ag


def get_env_params(env: gym.Env):
    obs = env.reset()

    params = {
        "obs": obs["observation"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
        "max_timesteps": env._max_episode_steps,
        "goal": obs["desired_goal"].shape[0],
        "action_space": env.action_space,
    }

    return params
