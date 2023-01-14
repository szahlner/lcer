import itertools
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)


class StandardScaler:
    """
    Standardscaler to transform data to zero mean and unit variance
    """

    def __init__(self) -> None:
        self.mu = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        """
        Runs two operations, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation

        :param data: A numpy array containing the input
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms the input matrix data using the parameters of this scaler

        :param data: A numpy array containing the points to be transformed
        :return: The transformed dataset
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Undoes the transformation performed by this scaler

        :param data: A numpy array containing the points to be transformed
        :return: The transformed dataset
        """
        return self.std * data + self.mu


def init_weights(m: Any) -> None:
    """
    Initialize weights and biases of layers of a neural network

    :param m: A layer of a neural network
    """

    def truncated_normal_init(t: torch.Tensor, mean: float = 0.0, std: float = 0.01):
        """
        Returns a truncated normal

        :param t:  An n-dimensional torch.Tensor
        :param mean: The mean of the normal distribution
        :param std: The standard deviation of the normal distribution
        :return:
        """
        torch.nn.init.normal_(t, mean=mean, std=std)

        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)

        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    """
    Fully Connected Ensemble layer

    :param in_features: Dimensions of input features
    :param out_features: Dimensions of output features
    :param ensemble_size: Ensemble size
    :param weight_decay: Weight decay to use
    :param bias: Use bias or not
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        weight_decay: float = 0.0,
        bias: bool = True,
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(self.in_features, self.out_features, self.bias is not None)


class EnsembleModel(nn.Module):
    """
    Ensemble Model as a combination of EnsembleFC layers combined in a neural network

    :param state_size: Dimensions of the state
    :param action_size: Dimensions of the action
    :param reward_size: Dimensions of the reward
    :param ensemble_size: Ensemble size
    :param hidden_size: Hidden layers in the ensemble
    :param learning_rate: Learning rate
    :param use_decay: Use weight decay or not
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        reward_size: int,
        ensemble_size: int,
        hidden_size: int = 200,
        learning_rate: float = 1e-3,
        use_decay: bool = False,
    ) -> None:
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size

        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_log_var = nn.Parameter(
            (torch.ones((1, self.output_dim)).float() / 2).to(device),
            requires_grad=False,
        )
        self.min_log_var = nn.Parameter(
            (-torch.ones((1, self.output_dim)).float() * 10).to(device),
            requires_grad=False,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()

    def forward(self, x: torch.Tensor, ret_log_var: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, : self.output_dim]

        log_var = self.max_log_var - F.softplus(self.max_log_var - nn5_output[:, :, self.output_dim :])
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)

        if ret_log_var:
            return mean, log_var
        else:
            return mean, torch.exp(log_var)

    def get_decay_loss(self) -> float:
        decay_loss = 0.0
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.0
        return decay_loss

    @staticmethod
    def loss(
        mean: torch.Tensor,
        log_var: torch.Tensor,
        labels: torch.Tensor,
        inc_var_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the loss

        :param mean: Ensemble_size x N x dim
        :param log_var: Ensemble_size x N x dim
        :param labels: N x dim
        :param inc_var_loss: Use var_loss or not
        :return: Total loss and MSE loss
        """
        assert len(mean.shape) == len(log_var.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(log_var, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train_model(self, loss: torch.Tensor) -> None:
        """
        Make the update step

        :param loss: Loss to be used in the update step
        """
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)

        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()

        self.optimizer.step()


class EnsembleDynamicsModel:
    """
    Ensemble Dynamics Model as a combination of EnsembleModel EnsembleFC layers combined in a neural network

    :param network_size: Ensemble size
    :param elite_size: Number of elites to be used
    :param state_size: Dimensions of the state
    :param action_size: Dimensions of the action
    :param reward_size: Dimensions of the reward
    :param hidden_size: Hidden layers in the ensemble
    :param use_decay: Use weight decay or not
    """

    def __init__(
        self,
        network_size: int,
        elite_size: int,
        state_size: int,
        action_size: int,
        reward_size: int = 1,
        hidden_size: int = 200,
        use_decay: bool = False,
    ) -> None:
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(
            state_size,
            action_size,
            reward_size,
            network_size,
            hidden_size,
            use_decay=use_decay,
        )
        self.scaler = StandardScaler()

        self._max_epochs_since_update = None
        self._epochs_since_update = None
        self._state = None
        self._snapshots = None

    def train(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 256,
        holdout_ratio: float = 0.0,
        max_epochs_since_update: int = 5,
    ) -> None:
        """
        Train the Ensemble Dynamics Model

        :param inputs: Inputs, normally state+actions
        :param labels: Labels or outputs normally rewards+next_states
        :param batch_size: Training batch size
        :param holdout_ratio: Train/Test split ratio
        :param max_epochs_since_update: How many training epochs before breaking the training
        """
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():

            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos : start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)

                self.ensemble_model.train_model(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(
                    holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False
                )
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[: self.elite_size].tolist()

                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

    def _save_best(self, epoch: int, holdout_losses: np.ndarray) -> bool:
        """
        Save the best trainings epoch in snapshots

        :param epoch: Current epoch number
        :param holdout_losses: Loss-value of holdouts
        :return: Whether there was an improvement in training or not
        """
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the rewards+next_states of a given input

        :param inputs: Inputs, normally states+actions
        :param batch_size: Batch size to process inputs
        :return: Ensemble mean and ensemble variance
        """
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i : min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        return ensemble_mean, ensemble_var


class Swish(nn.Module):
    """
    Swish activation function
    """

    def __init__(self) -> None:
        super(Swish, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x
