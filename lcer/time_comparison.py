import os
import pickle
import random
import time

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shadowhand_gym
import torch
from fast_pytorch_kmeans import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from lcer.mdp_dynamics_model.mdp_model import EnsembleDynamicsModel

matplotlib.use("Agg")
sns.set()

if torch.cuda.is_available():
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")


def pendulum() -> None:
    """
    Time test with the `InvertedPendulum-v2` environment.
    """

    ENV_ID = "InvertedPendulum-v2"
    SEED = 123
    MAX_NUMSTEPS = 50000
    STEPS = 1000
    RESOLUTION_STEPS = 5000
    K_NEIGHBORS = 10
    EXPERIMENT_PATH = "."
    N_EVAL = 20

    # ==================== Make env ====================
    env = gym.make(ENV_ID)

    # ==================== Seed ====================
    env.seed(SEED)
    env.action_space.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    obs = np.empty(shape=(MAX_NUMSTEPS, state_dim))
    actions = np.empty(shape=(MAX_NUMSTEPS, action_dim))
    rewards = np.empty(shape=(MAX_NUMSTEPS, 1))
    obs_next = np.empty_like(obs)

    total_numsteps = 0

    # ==================== Make dataset ====================
    while total_numsteps < MAX_NUMSTEPS:
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and total_numsteps < MAX_NUMSTEPS:
            action = env.action_space.sample()  # Sample random action

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1

            obs[total_numsteps] = state
            actions[total_numsteps] = action
            rewards[total_numsteps] = reward
            obs_next[total_numsteps] = next_state
            state = next_state

            total_numsteps += 1

    steps = STEPS
    eval_at_timesteps = [n * steps + steps for n in range(int(MAX_NUMSTEPS / steps))]
    nn_time_array = np.empty(shape=(len(eval_at_timesteps), N_EVAL))
    mk_time_array = np.empty_like(nn_time_array)
    k_time_array = np.empty_like(nn_time_array)
    mbpo_time_array = np.empty_like(nn_time_array)
    n_clusters = steps

    for n in range(N_EVAL):
        scaler, scaler_k = StandardScaler(), StandardScaler()
        kmeans_gpu = KMeans(n_clusters=n_clusters, mode="euclidean")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=2048, reassignment_ratio=0)
        clusters = [StandardScaler() for _ in range(n_clusters)]
        clusters_gpu = [[] for _ in range(n_clusters)]

        env_model = EnsembleDynamicsModel(
            network_size=7,
            elite_size=5,
            state_size=state_dim,
            action_size=action_dim,
            reward_size=1,
            hidden_size=200,
            use_decay=True,
        )

        for k, eval_timesteps in enumerate(eval_at_timesteps):
            # ==================== Nearest neighbor ====================
            start = time.time()
            # Construct Z-space
            z_space = np.concatenate((obs[:eval_timesteps], actions[:eval_timesteps]), axis=-1)
            z_space_norm = StandardScaler(with_mean=False).fit_transform(z_space)

            # NearestNeighbors - object
            k_nn = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(z_space_norm)
            nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

            end = time.time()
            nn_time_array[k, n] = end - start

            # ==================== Minibatch KMeans ====================
            if k == 0:
                o, a = obs[:eval_timesteps], actions[:eval_timesteps]
                r, o_2 = rewards[:eval_timesteps], obs_next[:eval_timesteps]
            else:
                o, a = obs[eval_at_timesteps[k - 1] : eval_timesteps], actions[eval_at_timesteps[k - 1] : eval_timesteps]
                r, o_2 = (
                    rewards[eval_at_timesteps[k - 1] : eval_timesteps],
                    obs_next[eval_at_timesteps[k - 1] : eval_timesteps],
                )

            start = time.time()
            # Construct Z-space
            z_space = np.concatenate((o, a), axis=-1)
            scaler.partial_fit(z_space)
            z_space_norm = scaler.transform(z_space)
            kmeans = kmeans.partial_fit(z_space_norm)

            # max startup steps, else max max_timesteps
            labels = kmeans.labels_
            z_space = np.concatenate((z_space, r, o_2), axis=-1)
            for m in range(len(labels)):
                clusters[labels[m]] = clusters[labels[m]].partial_fit(z_space[m].reshape(1, -1))

            end = time.time()
            mk_time_array[k, n] = end - start

            # ==================== KMeans (GPU) ====================
            start = time.time()

            z_space = np.concatenate((obs[:eval_timesteps], actions[:eval_timesteps]), axis=-1)
            scaler_k.partial_fit(z_space)
            z_space_norm = scaler_k.transform(z_space)
            z_space_norm = torch.tensor(z_space_norm, dtype=torch.float, device=torch_device)
            labels = kmeans_gpu.fit_predict(z_space_norm)
            labels = labels.detach().cpu().numpy()

            for m in range(n_clusters):
                buffer_idx = np.argwhere(labels == m)
                buffer_idx = buffer_idx.squeeze().tolist()
                if not isinstance(buffer_idx, list):
                    buffer_idx = [buffer_idx]

                clusters_gpu[n] = buffer_idx

            end = time.time()
            k_time_array[k, n] = end - start

            # ==================== MBPO ====================
            start = time.time()

            for _ in range(1):
                # Difference
                o, a = obs[:eval_timesteps], actions[:eval_timesteps]
                r, o_2 = rewards[:eval_timesteps], obs_next[:eval_timesteps]
                d_o = o_2 - o
                inputs = np.concatenate((o, a), axis=-1)
                labels = np.concatenate((r, d_o), axis=-1)

                # Train the environment model
                env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

            end = time.time()
            mbpo_time_array[k, n] = end - start

            print(f"{ENV_ID} - k: {k}")

        print(f"{ENV_ID} - n: {n}")

    nn_time_median = np.median(nn_time_array, axis=-1)
    mk_time_median = np.median(mk_time_array, axis=-1)
    k_time_median = np.median(k_time_array, axis=-1)
    mbpo_time_median = np.median(mbpo_time_array, axis=-1)

    # ==================== Plot ====================
    x_axis = np.arange(len(eval_at_timesteps))
    x_ticks_labels = []
    for t in eval_at_timesteps:
        if t % RESOLUTION_STEPS == 0:
            x_ticks_labels.append(f"{t}")
        else:
            x_ticks_labels.append("")

    plt.figure(ENV_ID, figsize=(9.60, 3.00))
    plt.plot(x_axis, mk_time_median, linewidth=1.5, label="Minibatch k-Means")
    plt.plot(x_axis, nn_time_median, linewidth=1.5, label="Nearest Neighbors")
    plt.plot(x_axis, k_time_median, linewidth=1.5, label="k-Means (GPU)")
    plt.plot(x_axis, mbpo_time_median, linewidth=1.5, label="k-Means (GPU)")
    plt.title(ENV_ID)
    plt.ylabel("Median time in seconds")
    plt.xticks(x_axis, x_ticks_labels, rotation=0)
    plt.xlabel("Timesteps")
    plt.legend(framealpha=0)
    plt.tight_layout()

    # save global plot
    save_path = os.path.join(EXPERIMENT_PATH, ENV_ID)
    plt.savefig(f"{save_path}.png", format="png")
    plt.savefig(f"{save_path}.pdf", format="pdf")

    data = {
        "nn_time_array": nn_time_array,
        "mk_time_array": mk_time_array,
        "k_time_array": k_time_array,
        "mbpo_time_array": mbpo_time_array,
        "eval_at_timesteps": eval_at_timesteps,
        "resolution_steps": RESOLUTION_STEPS,
        "env_id": ENV_ID,
    }
    with open(f"{save_path}.pkl", "wb") as file:
        pickle.dump(data, file)
        file.close()


def hopper() -> None:
    """
    Time test with the `Hopper-v2` environment.
    """

    ENV_ID = "Hopper-v2"
    SEED = 123
    MAX_NUMSTEPS = 50000
    STEPS = 1000
    RESOLUTION_STEPS = 5000
    K_NEIGHBORS = 10
    EXPERIMENT_PATH = "."
    N_EVAL = 20

    # ==================== Make env ====================
    env = gym.make(ENV_ID)

    # ==================== Seed ====================
    env.seed(SEED)
    env.action_space.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    state_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    obs = np.empty(shape=(MAX_NUMSTEPS, state_dim))
    actions = np.empty(shape=(MAX_NUMSTEPS, action_dim))
    rewards = np.empty(shape=(MAX_NUMSTEPS, 1))
    obs_next = np.empty_like(obs)

    total_numsteps = 0

    # ==================== Make dataset ====================
    while total_numsteps < MAX_NUMSTEPS:
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and total_numsteps < MAX_NUMSTEPS:
            action = env.action_space.sample()  # Sample random action

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1

            obs[total_numsteps] = state
            actions[total_numsteps] = action
            rewards[total_numsteps] = reward
            obs_next[total_numsteps] = next_state
            state = next_state

            total_numsteps += 1

    steps = STEPS
    eval_at_timesteps = [n * steps + steps for n in range(int(MAX_NUMSTEPS / steps))]
    nn_time_array = np.empty(shape=(len(eval_at_timesteps), N_EVAL))
    mk_time_array = np.empty_like(nn_time_array)
    k_time_array = np.empty_like(nn_time_array)
    mbpo_time_array = np.empty_like(nn_time_array)
    n_clusters = steps

    for n in range(N_EVAL):
        scaler, scaler_k = StandardScaler(), StandardScaler()
        kmeans_gpu = KMeans(n_clusters=n_clusters, mode="euclidean")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=2048, reassignment_ratio=0)
        clusters = [StandardScaler() for _ in range(n_clusters)]
        clusters_gpu = [[] for _ in range(n_clusters)]

        env_model = EnsembleDynamicsModel(
            network_size=7,
            elite_size=5,
            state_size=state_dim,
            action_size=action_dim,
            reward_size=1,
            hidden_size=200,
            use_decay=True,
        )

        for k, eval_timesteps in enumerate(eval_at_timesteps):
            # ==================== Nearest neighbor ====================
            start = time.time()
            # Construct Z-space
            z_space = np.concatenate((obs[:eval_timesteps], actions[:eval_timesteps]), axis=-1)
            z_space_norm = StandardScaler(with_mean=False).fit_transform(z_space)

            # NearestNeighbors - object
            k_nn = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(z_space_norm)
            nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

            end = time.time()
            nn_time_array[k, n] = end - start

            # ==================== Minibatch KMeans ====================
            if k == 0:
                o, a = obs[:eval_timesteps], actions[:eval_timesteps]
                r, o_2 = rewards[:eval_timesteps], obs_next[:eval_timesteps]
            else:
                o, a = obs[eval_at_timesteps[k - 1] : eval_timesteps], actions[eval_at_timesteps[k - 1] : eval_timesteps]
                r, o_2 = (
                    rewards[eval_at_timesteps[k - 1] : eval_timesteps],
                    obs_next[eval_at_timesteps[k - 1] : eval_timesteps],
                )

            start = time.time()
            # Construct Z-space
            z_space = np.concatenate((o, a), axis=-1)
            scaler.partial_fit(z_space)
            z_space_norm = scaler.transform(z_space)
            kmeans = kmeans.partial_fit(z_space_norm)

            # max startup steps, else max max_timesteps
            labels = kmeans.labels_
            z_space = np.concatenate((z_space, r, o_2), axis=-1)
            for m in range(len(labels)):
                clusters[labels[m]] = clusters[labels[m]].partial_fit(z_space[m].reshape(1, -1))

            end = time.time()
            mk_time_array[k, n] = end - start

            # ==================== KMeans (GPU) ====================
            start = time.time()

            z_space = np.concatenate((obs[:eval_timesteps], actions[:eval_timesteps]), axis=-1)
            scaler_k.partial_fit(z_space)
            z_space_norm = scaler_k.transform(z_space)
            z_space_norm = torch.tensor(z_space_norm, dtype=torch.float, device=torch_device)
            labels = kmeans_gpu.fit_predict(z_space_norm)
            labels = labels.detach().cpu().numpy()

            for m in range(n_clusters):
                buffer_idx = np.argwhere(labels == m)
                buffer_idx = buffer_idx.squeeze().tolist()
                if not isinstance(buffer_idx, list):
                    buffer_idx = [buffer_idx]

                clusters_gpu[n] = buffer_idx

            end = time.time()
            k_time_array[k, n] = end - start

            # ==================== MBPO ====================
            start = time.time()

            for _ in range(1):
                # Difference
                o, a = obs[:eval_timesteps], actions[:eval_timesteps]
                r, o_2 = rewards[:eval_timesteps], obs_next[:eval_timesteps]
                d_o = o_2 - o
                inputs = np.concatenate((o, a), axis=-1)
                labels = np.concatenate((r, d_o), axis=-1)

                # Train the environment model
                env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

            end = time.time()
            mbpo_time_array[k, n] = end - start

            print(f"{ENV_ID} - k: {k}")

        print(f"{ENV_ID} - n: {n}")

    nn_time_median = np.median(nn_time_array, axis=-1)
    mk_time_median = np.median(mk_time_array, axis=-1)
    k_time_median = np.median(k_time_array, axis=-1)
    mbpo_time_median = np.median(mbpo_time_array, axis=-1)

    # ==================== Plot ====================
    x_axis = np.arange(len(eval_at_timesteps))
    x_ticks_labels = []
    for t in eval_at_timesteps:
        if t % RESOLUTION_STEPS == 0:
            x_ticks_labels.append(f"{t}")
        else:
            x_ticks_labels.append("")

    plt.figure(ENV_ID, figsize=(9.60, 3.00))
    plt.plot(x_axis, mk_time_median, linewidth=1.5, label="Minibatch k-Means")
    plt.plot(x_axis, nn_time_median, linewidth=1.5, label="Nearest Neighbors")
    plt.plot(x_axis, k_time_median, linewidth=1.5, label="k-Means (GPU)")
    plt.plot(x_axis, mbpo_time_median, linewidth=1.5, label="k-Means (GPU)")
    plt.title(ENV_ID)
    plt.ylabel("Median time in seconds")
    plt.xticks(x_axis, x_ticks_labels, rotation=0)
    plt.xlabel("Timesteps")
    plt.legend(framealpha=0)
    plt.tight_layout()

    # save global plot
    save_path = os.path.join(EXPERIMENT_PATH, ENV_ID)
    plt.savefig(f"{save_path}.png", format="png")
    plt.savefig(f"{save_path}.pdf", format="pdf")

    data = {
        "nn_time_array": nn_time_array,
        "mk_time_array": mk_time_array,
        "k_time_array": k_time_array,
        "mbpo_time_array": mbpo_time_array,
        "eval_at_timesteps": eval_at_timesteps,
        "resolution_steps": RESOLUTION_STEPS,
        "env_id": ENV_ID,
    }
    with open(f"{save_path}.pkl", "wb") as file:
        pickle.dump(data, file)
        file.close()


def hand():
    """
    Time test with the `ShadowHandReach-v1` environment.
    """

    ENV_ID = "ShadowHandReach-v1"
    SEED = 123
    MAX_NUMSTEPS = 50000
    STEPS = 1000
    RESOLUTION_STEPS = 5000
    K_NEIGHBORS = 10
    EXPERIMENT_PATH = "."
    N_EVAL = 20

    # ==================== Make env ====================
    env = gym.make(ENV_ID)

    # ==================== Seed ====================
    env.seed(SEED)
    env.action_space.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    state_dim = np.prod(env.observation_space["observation"].shape)
    goal_dim = np.prod(env.observation_space["desired_goal"].shape)
    action_dim = np.prod(env.action_space.shape)
    obs = np.empty(shape=(MAX_NUMSTEPS, state_dim))
    goals = np.empty(shape=(MAX_NUMSTEPS, goal_dim))
    achieved_goals = np.empty_like(goals)
    achieved_goals_next = np.empty_like(goals)
    actions = np.empty(shape=(MAX_NUMSTEPS, action_dim))
    rewards = np.empty(shape=(MAX_NUMSTEPS, 1))
    obs_next = np.empty_like(obs)

    total_numsteps = 0

    # ==================== Make dataset ====================
    while total_numsteps < MAX_NUMSTEPS:
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and total_numsteps < MAX_NUMSTEPS:
            action = env.action_space.sample()  # Sample random action

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1

            obs[total_numsteps] = state["observation"]
            achieved_goals[total_numsteps] = state["achieved_goal"]
            goals[total_numsteps] = state["desired_goal"]
            actions[total_numsteps] = action
            rewards[total_numsteps] = reward
            obs_next[total_numsteps] = next_state["observation"]
            achieved_goals_next[total_numsteps] = next_state["achieved_goal"]
            state = next_state

            total_numsteps += 1

    steps = STEPS
    eval_at_timesteps = [n * steps + steps for n in range(int(MAX_NUMSTEPS / steps))]
    nn_time_array = np.empty(shape=(len(eval_at_timesteps), N_EVAL))
    mk_time_array = np.empty_like(nn_time_array)
    k_time_array = np.empty_like(nn_time_array)
    mbpo_time_array = np.empty_like(nn_time_array)
    n_clusters = steps

    for n in range(N_EVAL):
        scaler, scaler_k = StandardScaler(), StandardScaler()
        kmeans_gpu = KMeans(n_clusters=n_clusters, mode="euclidean")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=2048, reassignment_ratio=0)
        clusters = [StandardScaler() for _ in range(n_clusters)]
        clusters_gpu = [[] for _ in range(n_clusters)]

        env_model = EnsembleDynamicsModel(
            network_size=7,
            elite_size=5,
            state_size=state_dim,
            action_size=action_dim,
            reward_size=1,
            hidden_size=200,
            use_decay=True,
        )

        for k, eval_timesteps in enumerate(eval_at_timesteps):
            # ==================== Nearest neighbor ====================
            start = time.time()
            # Construct Z-space
            z_space = np.concatenate((obs[:eval_timesteps], goals[:eval_timesteps], actions[:eval_timesteps]), axis=-1)
            z_space_norm = StandardScaler(with_mean=False).fit_transform(z_space)

            # NearestNeighbors - object
            k_nn = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(z_space_norm)
            nn_indices = k_nn.kneighbors(z_space_norm, return_distance=False)

            end = time.time()
            nn_time_array[k, n] = end - start

            # ==================== Minibatch KMeans ====================
            if k == 0:
                o, g, a = obs[:eval_timesteps], goals[:eval_timesteps], actions[:eval_timesteps]
                r, o_2 = rewards[:eval_timesteps], obs_next[:eval_timesteps]
                ag, ag_2 = achieved_goals[:eval_timesteps], achieved_goals_next[:eval_timesteps]
            else:
                o, g, a = (
                    obs[eval_at_timesteps[k - 1] : eval_timesteps],
                    goals[eval_at_timesteps[k - 1] : eval_timesteps],
                    actions[eval_at_timesteps[k - 1] : eval_timesteps],
                )
                r, o_2 = (
                    rewards[eval_at_timesteps[k - 1] : eval_timesteps],
                    obs_next[eval_at_timesteps[k - 1] : eval_timesteps],
                )
                ag, ag_2 = (
                    achieved_goals[eval_at_timesteps[k - 1] : eval_timesteps],
                    achieved_goals_next[eval_at_timesteps[k - 1] : eval_timesteps],
                )

            start = time.time()
            # Construct Z-space
            z_space = np.concatenate((o, g, a), axis=-1)
            scaler.partial_fit(z_space)
            z_space_norm = scaler.transform(z_space)
            kmeans = kmeans.partial_fit(z_space_norm)

            # max startup steps, else max max_timesteps
            labels = kmeans.labels_
            z_space = np.concatenate((z_space, ag, o_2, ag_2), axis=-1)
            for m in range(len(labels)):
                clusters[labels[m]] = clusters[labels[m]].partial_fit(z_space[m].reshape(1, -1))

            end = time.time()
            mk_time_array[k, n] = end - start

            # ==================== KMeans (GPU) ====================
            start = time.time()

            z_space = np.concatenate((obs[:eval_timesteps], goals[:eval_timesteps], actions[:eval_timesteps]), axis=-1)
            scaler_k.partial_fit(z_space)
            z_space_norm = scaler_k.transform(z_space)
            z_space_norm = torch.tensor(z_space_norm, dtype=torch.float, device=torch_device)
            labels = kmeans_gpu.fit_predict(z_space_norm)
            labels = labels.detach().cpu().numpy()

            for m in range(n_clusters):
                buffer_idx = np.argwhere(labels == m)
                buffer_idx = buffer_idx.squeeze().tolist()
                if not isinstance(buffer_idx, list):
                    buffer_idx = [buffer_idx]

                clusters_gpu[n] = buffer_idx

            end = time.time()
            k_time_array[k, n] = end - start

            # ==================== MBPO ====================
            start = time.time()

            for _ in range(1):
                # Difference
                o, a = obs[:eval_timesteps], actions[:eval_timesteps]
                r, o_2 = rewards[:eval_timesteps], obs_next[:eval_timesteps]
                d_o = o_2 - o
                inputs = np.concatenate((o, a), axis=-1)
                labels = np.concatenate((r, d_o), axis=-1)

                # Train the environment model
                env_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

            end = time.time()
            mbpo_time_array[k, n] = end - start

            print(f"{ENV_ID} - k: {k}")

        print(f"{ENV_ID} - n: {n}")

    nn_time_median = np.median(nn_time_array, axis=-1)
    mk_time_median = np.median(mk_time_array, axis=-1)
    k_time_median = np.median(k_time_array, axis=-1)
    mbpo_time_median = np.median(mbpo_time_array, axis=-1)

    # ==================== Plot ====================
    x_axis = np.arange(len(eval_at_timesteps))
    x_ticks_labels = []
    for t in eval_at_timesteps:
        if t % RESOLUTION_STEPS == 0:
            x_ticks_labels.append(f"{t}")
        else:
            x_ticks_labels.append("")

    plt.figure(ENV_ID, figsize=(9.60, 3.00))
    plt.plot(x_axis, mk_time_median, linewidth=1.5, label="Minibatch k-Means (LCER-CC)")
    plt.plot(x_axis, k_time_median, linewidth=1.5, label="k-Means (GPU) (LCER-RM)")
    plt.plot(x_axis, nn_time_median, linewidth=1.5, label="Nearest Neighbors (NMER)")
    plt.plot(x_axis, mbpo_time_median, linewidth=1.5, label="fit MDP dynamics model (MBPO)")
    plt.title(ENV_ID)
    plt.ylabel("Median time in seconds")
    plt.xticks(x_axis, x_ticks_labels, rotation=0)
    plt.xlabel("Timesteps")
    plt.legend(framealpha=0)
    plt.tight_layout()

    # save global plot
    save_path = os.path.join(EXPERIMENT_PATH, ENV_ID)
    plt.savefig(f"{save_path}.png", format="png")
    plt.savefig(f"{save_path}.pdf", format="pdf")

    data = {
        "nn_time_array": nn_time_array,
        "mk_time_array": mk_time_array,
        "k_time_array": k_time_array,
        "mbpo_time_array": mbpo_time_array,
        "eval_at_timesteps": eval_at_timesteps,
        "resolution_steps": RESOLUTION_STEPS,
        "env_id": ENV_ID,
    }
    with open(f"{save_path}.pkl", "wb") as file:
        pickle.dump(data, file)
        file.close()


if __name__ == "__main__":
    pendulum()
    hopper()
    hand()
