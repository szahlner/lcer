import argparse
import datetime
import itertools
import json
import os
import random
import time
from collections import namedtuple

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lcer import HER, SAC
from lcer.common.arguments import ALGOS, POLICIES
from lcer.common.env_checker import check_env
from lcer.common.utils import get_env_params, get_predicted_states_her


def train_sac(parser: argparse.ArgumentParser) -> None:
    # Common arguments
    parser.add_argument("--lr", help="Learning rate for actor and critic", default=0.0003, type=float)
    parser.add_argument("--num-steps", help="Maximum number of steps", default=125001, type=int)
    parser.add_argument("--updates-per-step", help="Policy updates per environment step", default=1, type=int)

    args = parser.parse_args()

    # ==================== Environments ====================
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)
    if args.env_name == "Ant-v2":
        from lcer.common.envs.wrapper.ant_truncated_wrapper import AntTruncatedV2Wrapper

        env = AntTruncatedV2Wrapper(env)
        eval_env = AntTruncatedV2Wrapper(eval_env)

    # Check environments
    check_env(env)
    check_env(eval_env)

    # ==================== Seeding ====================
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env.seed(args.seed)
    eval_env.action_space.seed(args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ==================== Model-based ====================
    if args.model_based:
        from lcer.common.utils import get_predicted_states

        state_size = int(np.prod(env.observation_space.shape))
        action_size = int(np.prod(env.action_space.shape))

        if args.deterministic_model:
            from lcer.mdp_dynamics_model import DeterministicEnsembleDynamicsModel

            env_model = DeterministicEnsembleDynamicsModel(
                network_size=7,
                elite_size=5,
                state_size=state_size,
                action_size=action_size,
                reward_size=1,
                hidden_size=200,
                dropout_rate=0.05,
                use_decay=True,
            )
        else:
            from lcer.mdp_dynamics_model import StochasitcEnsembleDynamicsModel

            env_model = StochasitcEnsembleDynamicsModel(
                network_size=7,
                elite_size=5,
                state_size=state_size,
                action_size=action_size,
                reward_size=1,
                hidden_size=200,
                use_decay=True,
            )

    # ==================== Agent ====================
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    if args.save_agent:
        last_avg_reward_eval = None

    # ==================== Tensorboard ====================
    writer = SummaryWriter(
        "runs/{}_SAC_{}_{}_{}{}{}{}{}{}{}_vr{}_ur{}{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env_name,
            args.policy,
            args.seed,
            "_autotune" if args.automatic_entropy_tuning else "",
            "_mb" if args.model_based else "",
            "_nmer" if args.nmer else "",
            "_per" if args.per else "",
            "_lcercc" if args.lcercc else "",
            "_lcerrm" if args.lcerrm else "",
            args.v_ratio,
            args.updates_per_step,
            "_deterministic" if args.deterministic_model else "",
        )
    )

    # ==================== Save args/config to file ====================
    config_path = os.path.join(writer.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # ==================== Experience and memory ====================
    Experience = namedtuple("Experience", field_names="state action reward next_state mask")

    if args.model_based:
        if args.nmer:
            from lcer.common.replay_memory import MbpoNmerReplayMemory

            memory = MbpoNmerReplayMemory(
                args.replay_size,
                args.seed,
                v_ratio=args.v_ratio,
                env_name=args.env_name,
                args=args,
                k_neighbors=args.k_neighbors,
            )
        else:
            from lcer.common.replay_memory import MbpoReplayMemory

            memory = MbpoReplayMemory(args.replay_size, args.seed, v_ratio=args.v_ratio, env_name=args.env_name, args=args)
    else:
        if args.nmer and args.per:
            from lcer.common.replay_memory import PerNmerReplayMemory

            state_size = int(np.prod(env.observation_space.shape))
            action_size = int(np.prod(env.action_space.shape))
            memory = PerNmerReplayMemory(
                args.replay_size,
                args.seed,
                state_dim=state_size,
                action_dim=action_size,
                env_name=args.env_name,
                k_neighbors=args.k_neighbors,
            )
        elif args.nmer:
            from lcer.common.replay_memory import NmerReplayMemory

            memory = NmerReplayMemory(args.replay_size, args.seed, env_name=args.env_name, k_neighbors=args.k_neighbors)
        elif args.per:
            from lcer.common.replay_memory import PerReplayMemory

            state_size = int(np.prod(env.observation_space.shape))
            action_size = int(np.prod(env.action_space.shape))
            memory = PerReplayMemory(args.replay_size, args.seed, state_dim=state_size, action_dim=action_size)
        elif args.lcercc:
            from lcer.common.replay_memory import LocalClusterExperienceReplayClusterCenter

            state_size = int(np.prod(env.observation_space.shape))
            memory = LocalClusterExperienceReplayClusterCenter(
                args.replay_size,
                args.seed,
                state_dim=state_size,
                action_space=env.action_space,
                env_name=args.env_name,
                args=args,
            )
        elif args.lcerrm:
            from lcer.common.replay_memory import LocalClusterExperienceReplayRandomMember

            state_size = int(np.prod(env.observation_space.shape))
            memory = LocalClusterExperienceReplayRandomMember(
                args.replay_size,
                args.seed,
                state_dim=state_size,
                action_space=env.action_space,
                env_name=args.env_name,
                args=args,
            )
        else:
            from lcer.common.replay_memory import ReplayMemory

            memory = ReplayMemory(args.replay_size, args.seed)

    # ==================== Exploration loop ====================
    total_num_steps = 0

    while total_num_steps < args.start_steps:
        episode_trajectory = []
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and total_num_steps < args.start_steps:
            action = env.action_space.sample()  # Sample random action

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            episode_trajectory.append(Experience(state, action, reward, next_state, mask))
            state = next_state

        # Fill up replay memory
        steps_taken = len(episode_trajectory)
        # Normal experience replay
        for t in range(steps_taken):
            state, action, reward, next_state, mask = episode_trajectory[t]
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

    if args.lcercc or args.lcerrm:
        o = memory.buffer["state"][: len(memory)]
        a = memory.buffer["action"][: len(memory)]
        r = memory.buffer["reward"][: len(memory)]
        o_2 = memory.buffer["next_state"][: len(memory)]
        memory.update_clusters(o, a, r, o_2)

    if args.nmer:
        memory.update_neighbors()

    # ==================== Training loop ====================
    total_num_steps = 0
    updates = 0

    for n_episode in itertools.count(1):
        episode_trajectory = []
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        time_start = time.time()

        while not done:
            # Sample action from policy
            action = agent.select_action(state)

            if args.model_based and total_num_steps % args.update_env_model == 0:
                # Get real samples from environment
                batch_size = len(memory)
                o, a, r, o_2, _ = memory.sample_r(batch_size=batch_size)

                # Difference
                d_o = o_2 - o
                inputs = np.concatenate((o, a), axis=-1)
                labels = np.concatenate((np.expand_dims(r, axis=-1), d_o), axis=-1)

                # Train the environment model
                env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

                # Resize buffer capacity
                current_epoch = int(total_num_steps / args.epoch_length)
                memory.set_rollout_length(current_epoch)
                memory.resize_v_memory()

                # Rollout the environment model
                o, _, _, _, _ = memory.sample_r(batch_size=args.n_rollout_samples)

                for n in range(memory.rollout_length):
                    a = agent.select_action(o)
                    r, o_2, d = get_predicted_states(env_model, o, a, args.env_name)
                    # Push into memory
                    for k in range(len(o)):
                        memory.push_v(o[k], a[k], float(r[k]), o_2[k], float(not d[k]))
                    non_term_mask = ~d.squeeze(-1)
                    if non_term_mask.sum() == 0:
                        break
                    o = o_2[non_term_mask]

                if args.nmer:
                    memory.update_v_neighbors()

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    if args.per:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters_per(
                            memory, args.batch_size, updates
                        )
                    else:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                            memory, args.batch_size, updates
                        )
                    updates += 1

                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                writer.add_scalar("entropy_temprature/alpha", alpha, updates)

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            episode_trajectory.append(Experience(state, action, reward, next_state, mask))
            state = next_state

            if total_num_steps % args.eval_timesteps == 0 and args.eval:
                avg_reward_eval = 0.0
                episodes_eval = args.eval_episodes
                for _ in range(episodes_eval):
                    state_eval = eval_env.reset()
                    episode_reward_eval = 0
                    done_eval = False
                    while not done_eval:
                        action_eval = agent.select_action(state_eval, evaluate=True)

                        next_state_eval, reward_eval, done_eval, _ = eval_env.step(action_eval)
                        episode_reward_eval += reward_eval

                        state_eval = next_state_eval
                    avg_reward_eval += episode_reward_eval
                avg_reward_eval /= episodes_eval

                writer.add_scalar("avg_reward/test_timesteps", avg_reward_eval, total_num_steps)

                if args.lcercc or args.lcerrm:
                    memory.save_cluster_centers(total_num_steps, writer.log_dir)

                print("----------------------------------------")
                print("Timestep Eval - Test Episodes: {}, Avg. Reward: {}".format(episodes_eval, round(avg_reward_eval, 2)))
                print("----------------------------------------")

                if args.save_agent:
                    ckpts = []
                    for file in os.listdir(writer.log_dir):
                        if file.endswith(".zip"):
                            ckpts.append(os.path.join(writer.log_dir, file))
                    if len(ckpts) > args.keep_best_agents:
                        latest_ckpts = sorted(ckpts, key=os.path.getctime)
                        for ckpt in latest_ckpts[: -args.keep_best_agents]:
                            os.remove(ckpt)

                    if last_avg_reward_eval is None or avg_reward_eval > last_avg_reward_eval:
                        agent.save_checkpoint(args.env_name, writer.log_dir, total_num_steps)
                        last_avg_reward_eval = avg_reward_eval

        # Fill up replay memory
        steps_taken = len(episode_trajectory)
        # Normal experience replay
        for t in range(steps_taken):
            state, action, reward, next_state, mask = episode_trajectory[t]
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        if args.lcercc or args.lcerrm:
            o = memory.buffer["state"][total_num_steps - steps_taken : total_num_steps]
            a = memory.buffer["action"][total_num_steps - steps_taken : total_num_steps]
            r = memory.buffer["reward"][total_num_steps - steps_taken : total_num_steps]
            o_2 = memory.buffer["next_state"][total_num_steps - steps_taken : total_num_steps]
            memory.update_clusters(o, a, r, o_2)

        if args.nmer:
            memory.update_neighbors()

        writer.add_scalar("reward/train_timesteps", episode_reward, total_num_steps)
        print(
            "Episode: {}, total num steps: {}, episode steps: {}, reward: {}, time/step: {}s".format(
                n_episode,
                total_num_steps,
                episode_steps,
                round(episode_reward, 2),
                round((time.time() - time_start) / episode_steps, 3),
            )
        )

        if total_num_steps > args.num_steps:
            break

    env.close()


def train_sac_her(parser: argparse.ArgumentParser) -> None:
    # Common arguments
    parser.add_argument("--lr", help="Learning rate for actor and critic", default=0.001, type=float)
    parser.add_argument("--num-steps", help="Maximum number of steps", default=50001, type=int)
    parser.add_argument("--updates-per-step", help="Policy updates per environment step", default=0, type=int)

    # Hindsight Experience Replay (HER) arguments
    parser.add_argument("--her-replay-strategy", help="Replay strategy", default="future", type=str)
    parser.add_argument("--her-replay-k", help="Replay k, probability", default=4, type=int)
    parser.add_argument("--her", help="Whether to use Hindsight Experience Replay (HER) or not", action="store_true")
    parser.add_argument(
        "--her-normalize",
        help="Whether to use HER with normalization of observations and goals",
        action="store_true",
    )
    parser.add_argument("--n-update-batches", help="Updates per rollout", default=20, type=int)

    args = parser.parse_args()
    assert (
        args.updates_per_step > 0 and args.n_update_batches == 0 or args.updates_per_step == 0 and args.n_update_batches > 0
    ), "One of --updates-per-step or --n-update-batches must be zero (0)"

    # ==================== Environments ====================
    if "ShadowHandReach" in args.env_name or "ShadowHandBlock" in args.env_name:
        import shadowhand_gym

    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    # Check environments
    check_env(env)
    check_env(eval_env)

    env_params = get_env_params(env)

    # ==================== Seeding ====================
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env.seed(args.seed)
    eval_env.action_space.seed(args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ==================== Model-based ====================
    if args.model_based:
        from lcer.common.utils import get_predicted_states

        state_size = env_params["obs"] + 2 * env_params["goal"]
        action_size = env_params["action"]

        if args.deterministic_model:
            from lcer.mdp_dynamics_model import DeterministicEnsembleDynamicsModel

            env_model = DeterministicEnsembleDynamicsModel(
                network_size=7,
                elite_size=5,
                state_size=state_size,
                action_size=action_size,
                reward_size=1,
                hidden_size=200,
                dropout_rate=0.05,
                use_decay=True,
            )
        else:
            from lcer.mdp_dynamics_model import StochasitcEnsembleDynamicsModel

            env_model = StochasitcEnsembleDynamicsModel(
                network_size=7,
                elite_size=5,
                state_size=state_size,
                action_size=action_size,
                reward_size=1,
                hidden_size=200,
                use_decay=True,
            )

    # ==================== Agent ====================
    agent = SAC(env_params["obs"] + env_params["goal"], env.action_space, args)
    if args.save_agent:
        last_avg_reward_eval = None

    # ==================== Tensorboard ====================
    writer = SummaryWriter(
        "runs/{}_SAC_{}_{}_{}{}{}{}{}{}{}_vr{}_ur{}_nub{}{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            args.env_name,
            args.policy,
            args.seed,
            "_autotune" if args.automatic_entropy_tuning else "",
            "_mb" if args.model_based else "",
            "_nmer" if args.nmer else "",
            "_her" if args.her else "",
            "_lcercc" if args.lcercc else "",
            "_lcerrm" if args.lcerrm else "",
            args.v_ratio,
            args.updates_per_step,
            args.n_update_batches,
            "_deterministic" if args.deterministic_model else "",
        )
    )

    # ==================== Save args/config to file ====================
    config_path = os.path.join(writer.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # ==================== Experience and memory ====================
    Experience = namedtuple("Experience", field_names="state action reward next_state mask")

    if args.model_based:
        from lcer.common.her.replay_memory import HerMbpoReplayMemory, HerSampler

        sampler = HerSampler("future", args.her_replay_k, env.compute_reward)
        memory = HerMbpoReplayMemory(
            env_params,
            args.replay_size,
            v_ratio=args.v_ratio,
            args=args,
            sample_func=sampler.sample_her_transitions,
            normalize=args.her_normalize,
        )
    else:
        if args.her and not args.nmer:
            from lcer.common.her.replay_memory import HerReplayMemory, HerSampler

            sampler = HerSampler("future", args.her_replay_k, env.compute_reward)
            memory = HerReplayMemory(
                env_params, args.replay_size, sample_func=sampler.sample_her_transitions, normalize=args.her_normalize
            )
        elif args.nmer and args.her:
            from lcer.common.her.replay_memory import HerNmerReplayMemory

            memory = HerNmerReplayMemory(env_params, args.replay_size, args=args, normalize=args.her_normalize)
        elif args.nmer:
            from lcer.common.her.replay_memory import NmerReplayMemory

            memory = NmerReplayMemory(env_params, args.replay_size, args=args, normalize=args.her_normalize)
        elif args.lcercc:
            from lcer.common.her.replay_memory import HerLocalClusterExperienceReplayClusterCenterReplayMemory

            memory = HerLocalClusterExperienceReplayClusterCenterReplayMemory(
                env_params, args.replay_size, args=args, normalize=args.her_normalize
            )
        elif args.lcerrm:
            from lcer.common.her.replay_memory import HerLocalClusterExperienceReplayRandomMemberReplayMemory

            memory = HerLocalClusterExperienceReplayRandomMemberReplayMemory(
                env_params, args.replay_size, args=args, normalize=args.her_normalize
            )
        else:
            from lcer.common.her.replay_memory import SimpleReplayMemory

            memory = SimpleReplayMemory(env_params, args.replay_size, args=args, normalize=args.her_normalize)

    # ==================== Exploration loop ====================
    total_num_steps = 0

    while total_num_steps < args.start_steps:
        episode_trajectory = []
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and total_num_steps < args.start_steps:
            action = env.action_space.sample()  # Sample random action

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            episode_trajectory.append(Experience(state, action, reward, next_state, mask))
            state = next_state

        # Fill up replay memory
        steps_taken = len(episode_trajectory)
        o, ag, g, a = [], [], [], []

        # Normal experience replay
        for t in range(steps_taken):
            state, action, reward, next_state, mask = episode_trajectory[t]
            # Append transition to memory
            o.append(state["observation"]), ag.append(state["achieved_goal"])
            g.append(state["desired_goal"]), a.append(action)

        o.append(next_state["observation"]), ag.append(next_state["achieved_goal"])
        o, ag, g, a = np.array([o]), np.array([ag]), np.array([g]), np.array([a])
        memory.push_episode([o, ag, g, a])

    if args.lcercc or args.lcerrm:
        o = memory.buffers["obs"][: len(memory)]
        ag = memory.buffers["ag"][: len(memory)]
        a = memory.buffers["actions"][: len(memory)]
        o_2 = memory.buffers["obs_next"][: len(memory)]
        ag_2 = memory.buffers["ag_next"][: len(memory)]
        g = memory.buffers["g"][: len(memory)]
        memory.update_clusters(o, ag, a, o_2, ag_2, g)

    if args.nmer:
        memory.update_neighbors()

    # ==================== Training loop ====================
    total_num_steps = 0
    updates = 0

    for n_episode in itertools.count(1):
        episode_trajectory = []
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        time_start = time.time()

        while not done:
            if args.her_normalize:
                state_norm = memory.o_norm.normalize(state["observation"])
                goal_norm = memory.g_norm.normalize(state["desired_goal"])
                state_ = np.concatenate((state_norm, goal_norm), axis=-1)
            else:
                state_ = np.concatenate((state["observation"], state["desired_goal"]), axis=-1)
            action = agent.select_action(state_)  # Sample action from policy

            if args.model_based and total_num_steps % args.update_env_model == 0:
                # Get real samples from environment
                batch_size = max(len(memory), 10000)
                transitions = memory.sample_r(batch_size=batch_size, return_transitions=True)

                inputs = np.concatenate(
                    (transitions["obs"], transitions["ag"], transitions["g"], transitions["actions"]), axis=-1
                )
                # Difference
                labels = np.concatenate(
                    (transitions["obs_next"], transitions["ag_next"], transitions["g"]), axis=-1
                ) - np.concatenate((transitions["obs"], transitions["ag"], transitions["g"]), axis=-1)

                # Train the environment model
                env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

                # Resize buffer capacity
                current_epoch = int(total_num_steps / args.epoch_length)
                current_rollout_length = memory.rollout_length
                memory.set_rollout_length(current_epoch)
                if current_rollout_length != memory.rollout_length:
                    memory.resize_v_memory()

                # Rollout the environment model
                # o, o_ag, o_g, _, _, _, _, _, _ = memory.sample_r(batch_size=args.n_rollout_samples)
                transitions = memory.sample_r(batch_size=args.n_rollout_samples, return_transitions=True)
                o, o_ag, o_g = transitions["obs"], transitions["ag"], transitions["g"]

                # Preallocate
                v_state = np.empty(shape=(args.n_rollout_samples, memory.rollout_length + 1, env_params["obs"]))
                v_state_ag = np.empty(shape=(args.n_rollout_samples, memory.rollout_length + 1, env_params["goal"]))
                v_state_g = np.empty(shape=(args.n_rollout_samples, memory.rollout_length, env_params["goal"]))
                v_action = np.empty(shape=(args.n_rollout_samples, memory.rollout_length, env_params["action"]))

                # Rollout
                for n in range(memory.rollout_length):
                    if args.her_normalize:
                        o_norm = memory.o_norm.normalize(o)
                        g_norm = memory.g_norm.normalize(o_g)
                        o_ = np.concatenate((o_norm, g_norm), axis=-1)
                    else:
                        o_ = np.concatenate((o, o_g), axis=-1)
                    a = agent.select_action(o_)
                    o_2, o_2_ag = get_predicted_states_her(env_model, o, o_ag, o_g, a, env_params)
                    # Push into memory
                    v_state[:, n], v_state_ag[:, n], v_state_g[:, n], v_action[:, n] = o, o_ag, o_g, a
                    o, o_ag = o_2, o_2_ag
                v_state[:, -1], v_state_ag[:, -1] = o, o_ag

                for n in range(len(v_state)):
                    memory.push_v([v_state[n][None, :], v_state_ag[n][None, :], v_state_g[n][None, :], v_action[n][None, :]])

            if len(memory) > args.batch_size and args.updates_per_step > 0:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates
                    )
                    updates += 1
                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                writer.add_scalar("entropy_temprature/alpha", alpha, updates)

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_num_steps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            episode_trajectory.append(Experience(state, action, reward, next_state, mask))
            state = next_state

            if total_num_steps % args.eval_timesteps == 0 and args.eval:
                avg_reward_eval = 0.0
                episodes_eval = args.eval_episodes  # 10
                total_success_rate = []
                for _ in range(episodes_eval):
                    state_eval = eval_env.reset()
                    episode_reward_eval = 0
                    done_eval = False
                    per_success_rate = []
                    while not done_eval:
                        if args.her_normalize:
                            state_norm = memory.o_norm.normalize(state_eval["observation"])
                            goal_norm = memory.g_norm.normalize(state_eval["desired_goal"])
                            state_eval_ = np.concatenate((state_norm, goal_norm), axis=-1)
                        else:
                            state_eval_ = np.concatenate((state_eval["observation"], state_eval["desired_goal"]), axis=-1)
                        action_eval = agent.select_action(state_eval_, evaluate=True)

                        next_state_eval, reward_eval, done_eval, info = eval_env.step(action_eval)
                        episode_reward_eval += reward_eval
                        per_success_rate.append(info["is_success"])

                        state_eval = next_state_eval
                    avg_reward_eval += episode_reward_eval
                    total_success_rate.append(per_success_rate)

                avg_reward_eval /= episodes_eval
                total_success_rate = np.array(total_success_rate)
                total_success_rate = np.mean(total_success_rate[:, -1])

                writer.add_scalar("avg_reward/test_timesteps", avg_reward_eval, total_num_steps)
                writer.add_scalar("avg_reward/test_success_rate", total_success_rate, total_num_steps)

                if args.lcercc or args.lcerrm:
                    memory.save_cluster_centers(total_num_steps, writer.log_dir)

                print("----------------------------------------")
                print(
                    "Timestep Eval - Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(
                        episodes_eval, round(avg_reward_eval, 2), round(total_success_rate, 3)
                    )
                )
                print("----------------------------------------")

                if args.save_agent:
                    ckpts = []
                    for file in os.listdir(writer.log_dir):
                        if file.endswith(".zip"):
                            ckpts.append(os.path.join(writer.log_dir, file))
                    if len(ckpts) > args.keep_best_agents:
                        latest_ckpts = sorted(ckpts, key=os.path.getctime)
                        for ckpt in latest_ckpts[: -args.keep_best_agents]:
                            os.remove(ckpt)

                    if last_avg_reward_eval is None or avg_reward_eval > last_avg_reward_eval:
                        if args.her_normalize:
                            agent.save_checkpoint(args.env_name, writer.log_dir, total_num_steps, memory=memory)
                        else:
                            agent.save_checkpoint(args.env_name, writer.log_dir, total_num_steps)
                        last_avg_reward_eval = avg_reward_eval

        # Fill up replay memory
        steps_taken = len(episode_trajectory)

        o, ag, g, a = [], [], [], []

        # Normal experience replay
        for t in range(steps_taken):
            state, action, reward, next_state, mask = episode_trajectory[t]
            # Append transition to memory
            o.append(state["observation"]), ag.append(state["achieved_goal"])
            g.append(state["desired_goal"]), a.append(action)

        o.append(next_state["observation"]), ag.append(next_state["achieved_goal"])
        o, ag, g, a = np.array([o]), np.array([ag]), np.array([g]), np.array([a])
        memory.push_episode([o, ag, g, a])

        if args.lcercc or args.lcerrm:
            memory.update_clusters(o[0, :-1], ag[0, :-1], a[0], o[0, 1:], ag[0, 1:], g[0])

        if args.nmer:
            memory.update_neighbors()

        if len(memory) > args.batch_size and args.n_update_batches > 0:
            # Number of updates per step in environment
            for i in range(args.n_update_batches):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                    memory, args.batch_size, updates
                )
                updates += 1
            writer.add_scalar("loss/critic_1", critic_1_loss, updates)
            writer.add_scalar("loss/critic_2", critic_2_loss, updates)
            writer.add_scalar("loss/policy", policy_loss, updates)
            writer.add_scalar("loss/entropy_loss", ent_loss, updates)
            writer.add_scalar("entropy_temprature/alpha", alpha, updates)

        writer.add_scalar("reward/train_timesteps", episode_reward, total_num_steps)
        print(
            "Episode: {}, total num steps: {}, episode steps: {}, reward: {}, time/step: {}s".format(
                n_episode,
                total_num_steps,
                episode_steps,
                round(episode_reward, 2),
                round((time.time() - time_start) / episode_steps, 3),
            )
        )

        if total_num_steps > args.num_steps:
            break

    env.close()


def train() -> None:
    parser = argparse.ArgumentParser(description="Local Cluster Experience Replay (LCER) Trainings Script - Arguments")

    # Common arguments
    parser.add_argument("--algo", help="RL Algorithm", default="sac", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env-name", help="Environment ID", default="Hopper-v2", type=str)
    parser.add_argument("--policy", help="Policy type", default="Gaussian", type=str, required=False, choices=POLICIES)
    parser.add_argument("--no-eval", help="Whether to evaluate the policy or not", action="store_true")

    args = parser.parse_args()
    if args.no_eval:
        parser.add_argument("--eval", action="store_true")
    else:
        parser.add_argument("--eval", action="store_false")

    parser.add_argument("--gamma", help="Discount factor for reward", default=0.99, type=float)
    parser.add_argument("--tau", help="Target smoothing coefficient", default=0.005, type=float)
    parser.add_argument(
        "--alpha",
        help="Temperature parameter, determines the relative importance of the entropy term against the reward",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--no-automatic-entropy-tuning",
        help="Whether to automatically adjust alpha (temperature parameter) or not",
        action="store_false",
    )

    args = parser.parse_args()
    if args.no_automatic_entropy_tuning:
        parser.add_argument("--automatic-entropy-tuning", action="store_true")
    else:
        parser.add_argument("--automatic-entropy-tuning", action="store_false")

    parser.add_argument("--seed", help="Random seed", default=123456, type=int)
    parser.add_argument("--batch-size", help="Batch size", default=256, type=int)
    parser.add_argument("--hidden-size", help="Hidden size of actor and critic network", default=256, type=int)
    parser.add_argument("--start-steps", help="Steps sampling random actions", default=5000, type=int)
    parser.add_argument(
        "--target-update-interval",
        help="Value target update per number of updates per step",
        default=1,
        type=int,
    )
    parser.add_argument("--replay-size", help="Size of the replay-buffer", default=int(1e6), type=int)
    parser.add_argument("--target-entropy", help="Target entropy", default=-1, type=float)
    parser.add_argument("--eval-episodes", help="How many episodes to eval", default=10, type=int)
    parser.add_argument("--cuda", help="Whether to use CUDA or not", action="store_true")

    args = parser.parse_args()
    args.cuda = True if torch.cuda.is_available() else False

    parser.add_argument("--eval-timesteps", help="When to evaluate the policy", default=1000, type=int)
    parser.add_argument("--save-agent", help="Whether or not to save the agent", action="store_true")
    parser.add_argument("--keep-best-agents", help="Keep best X agents", default=10, type=int)

    # Model-Based Policy Optimization (MBPO) arguments
    parser.add_argument(
        "--model-based",
        help="Whether to use Model-Based Policy Optimization (MBPO) model-based RL or not",
        action="store_true",
    )
    parser.add_argument("--v-ratio", help="Virtual to real data ratio", default=0.95, type=float)
    parser.add_argument("--update-env-model", help="When to update the dynamics model, in timesteps", default=250, type=int)
    parser.add_argument(
        "--n-training-samples",
        help="Number of samples to train the dynamics model on",
        default=100000,
        type=int,
    )
    parser.add_argument(
        "--n-rollout-samples",
        help="Number of samples to rollout the dynamics model",
        default=100000,
        type=int,
    )
    parser.add_argument("--model-retain-epochs", help="How many epochs to retain", default=1, type=int)
    parser.add_argument("--epoch-length", help="Steps per epoch", default=1000, type=int)
    parser.add_argument("--rollout-min-epoch", help="Rollout starts from min epoch", default=20, type=int)
    parser.add_argument("--rollout-max-epoch", help="Rollout ends at max epoch", default=150, type=int)
    parser.add_argument("--rollout-min-length", help="Rollout has min length", default=1, type=int)
    parser.add_argument("--rollout-max-length", help="Rollout has max length", default=15, type=int)
    parser.add_argument("--deterministic-model", help="Use deterministic model-based model", action="store_true")

    # Neighborhood Mixup Experience Replay (NMER) arguments
    parser.add_argument(
        "--nmer",
        help="Whether to use Neighborhood Mixup Experience Replay (NMER) or not",
        action="store_true",
    )
    parser.add_argument("--k-neighbors", help="Amount of neighbors", default=10, type=int)

    # Prioritized Experience Replay (PER) arguments
    parser.add_argument("--per", help="Whether to use Prioritized Experience Replay (PER) or not", action="store_true")

    # Local Cluster Experience Replay (LCER) arguments
    parser.add_argument(
        "--lcercc",
        help="Whether to use Local Cluster Experience Replay Cluster Center (LCER-CC) or not",
        action="store_true",
    )
    parser.add_argument(
        "--lcerrm",
        help="Whether to use Local Cluster Experience Replay Random Member (LCER-RM) or not",
        action="store_true",
    )
    parser.add_argument("--n-clusters", help="Amount of clusters to use for LCER", default=-1, type=int)

    args = parser.parse_args()
    assert (
        args.lcercc and not args.lcerrm or not args.lcercc and args.lcerrm or not args.lcercc and not args.lcerrm
    ), "LCER-CC and LCER-RM must not be both active at the same time"

    args = parser.parse_args()
    if args.algo == "sac":
        train_sac(parser)


if __name__ == "__main__":
    train()
