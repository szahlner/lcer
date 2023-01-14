import argparse
import os
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch.optim import Adam

from lcer.common.replay_memory import (
    BaseReplayMemory,
    PerNmerReplayMemory,
    PerReplayMemory,
    ReplayMemory,
)
from lcer.common.utils import hard_update, soft_update
from lcer.sac.sac_model import DeterministicPolicy, GaussianPolicy, QNetwork


class SAC:
    """
    Soft Actor-Critic reinforcement learning agent

    :param num_inputs: Dimensions of state features
    :param action_space: Action space of the environment
    :param args: Arguments from command line
    """

    def __init__(self, num_inputs: int, action_space: spaces.Space, args: argparse.Namespace) -> None:
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":

            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if args.automatic_entropy_tuning:
                if args.target_entropy is not None:
                    self.target_entropy = args.target_entropy
                else:
                    self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                self.alpha = self.log_alpha.cpu().exp().item()
            else:

                # Alpha
                self.alpha = args.alpha

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device
            )
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action for a given state

        :param state: A state of an environment transition
        :param evaluate: Whether to evaluate or not (with or without noise)
        :return: The sampled action
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(
        self,
        memory: Union[ReplayMemory, BaseReplayMemory, PerReplayMemory, PerNmerReplayMemory],
        batch_size: int,
        updates: int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Update the neural network parameters

        :param memory: Replay-buffer to be used to sample the batch of transitions
        :param batch_size: How many transitions to sample in one batch
        :param updates: Current number of updates
        :return: Loss items from qf1, qf2, policy, alpha and alpha_tlogs
        """

        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
        ) = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    def update_parameters_per(
        self,
        memory: Union[PerReplayMemory, PerNmerReplayMemory],
        batch_size: int,
        updates: int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Update the neural network parameters

        :param memory: Replay-buffer to be used to sample the batch of transitions
        :param batch_size: How many transitions to sample in one batch
        :param updates: Current number of updates
        :return: Loss items from qf1, qf2, policy, alpha and alpha_tlogs
        """

        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            mask_batch,
            weights_batch,
            indices,
        ) = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)
        weights_batch = torch.FloatTensor(weights_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss_element_wise = (qf1 - next_q_value).pow(2)
        qf2_loss_element_wise = (qf2 - next_q_value).pow(2)
        qf_loss_element_wise = qf1_loss_element_wise + qf2_loss_element_wise
        qf_loss = (qf_loss_element_wise * weights_batch).mean()
        qf1_loss = (qf1_loss_element_wise * weights_batch).detach().mean()
        qf2_loss = (qf2_loss_element_wise * weights_batch).detach().mean()

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss_element_wise = (self.alpha * log_pi) - min_qf_pi
        policy_loss = (policy_loss_element_wise * weights_batch).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # PER: update priorities
        new_priorities = qf_loss_element_wise
        new_priorities += policy_loss_element_wise.pow(2)
        new_priorities += memory.priority_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        memory.update_priorities(indices, new_priorities)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    def save_checkpoint(self, env_name: str, ckpt_path: str, suffix: str = None) -> None:
        """
        Save model parameters

        :param env_name: Id/Name of the environment
        :param ckpt_path: Save path for the checkpoint
        :param suffix: Optional suffix (defaults to: "")
        """
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if suffix is None:
            file_name = "sac_ckpt_{}.zip".format(env_name)
        else:
            file_name = "sac_ckpt_{}_{}.zip".format(env_name, suffix)

        ckpt_path = os.path.join(ckpt_path, file_name)

        data = {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
        }
        if hasattr(self, "alpha_optim"):
            data["alpha"] = self.alpha
            data["log_alpha"] = self.log_alpha
            data["alpha_optimizer_state_dict"] = self.alpha_optim.state_dict()

        print("Saving models to {}".format(ckpt_path))
        torch.save(data, ckpt_path)

    def load_checkpoint(self, ckpt_path: str, evaluate: bool = False) -> None:
        """
        Load model parameters

        :param ckpt_path: Save path of the checkpoint
        :param evaluate: Whether to load in evaluation mode or training mode
        """

        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.policy.load_state_dict(ckpt["policy_state_dict"])
            self.critic.load_state_dict(ckpt["critic_state_dict"])
            self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
            self.critic_optim.load_state_dict(ckpt["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(ckpt["policy_optimizer_state_dict"])

            if hasattr(self, "alpha_optim"):
                self.alpha = ckpt["alpha"]
                self.log_alpha = ckpt["log_alpha"]
                self.alpha_optim.load_state_dict(ckpt["alpha_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
