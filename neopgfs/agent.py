import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import numpy as np
from typing import Tuple
from .replay_buffer import MinibatchSample
from torch.optim import Adam


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_T_dim: int, action_R_dim: int,
    ):
        """Generates a new actor object containing two neural networks for function
        approximation: f(state) -> action_T and pi(state, action_T) -> action_R.

        Args:
            state_dim (int): Dimension of state vector (n_bits)
            action_T_dim (int): Dimension of action_T vector (num_reactions)
            action_R_dim (int): Dimension of action_R vector (num_features)
        """
        super(Actor, self).__init__()

        # f Network to choose template f(state) -> action_T
        self.f = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_T_dim),
            nn.Tanh(),
        )

        # Deterministic policy architecture pi(state, T_one_hot) -> action
        self.pi = nn.Sequential(
            nn.Linear(state_dim + action_T_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 167),
            nn.ReLU(),
            nn.Linear(167, action_R_dim),
            nn.Tanh(),
        )

    def forward(
        self, state: Tensor, T_mask: Tensor, gumbel_tau: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Makes a forward pass for the actor, returns action_R and action_T.

        Args:
            state (Tensor): tensor representing current state
            T_mask (Tensor): one-hot tensor representing the reactions that might be used
            with current state as first reactant
            gumbel_tau (float): temperature parameter for Gumbel Softmax function,
            controlling the degree of exploration to be performed

        Returns:
            Tuple[Tensor, Tensor, Tensor]: action_T representing the chosen reaction template,
            action_R representing the chosen second reactant, and T representing the raw
            output of the f network
        """
        # Choose action_T using f network
        T_raw: Tensor = self.f(state)

        # Transform action_T to one_hot using Gumbel Softmax
        action_T = F.gumbel_softmax(T_raw * T_mask, gumbel_tau)

        # Choose action_R using policy network
        action_R: Tensor = self.pi(torch.cat([state, action_T]), axis=1)

        return action_T, action_R, T_raw


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_T_dim: int, action_R_dim: int):
        """Creates a critic object holding two neural networks representing
        two versions of the Q-value function: Q(state, action_T, action_R) -> float

        Args:
            state_dim (int): Dimension of the state vector (num_bits)
            action_T_dim (int): Dimension of action_T (num_reactions)
            action_R_dim (int): Dimension of action_R (num_descriptors)
        """
        super(Critic, self).__init__()

        self.Q1_model = nn.Sequential(
            nn.Linear(state_dim + action_T_dim + action_R_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.Q2_model = nn.Sequential(
            nn.Linear(state_dim + action_T_dim + action_R_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self, state: Tensor, action_T: Tensor, action_R: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Makes a forward pass of the critic object, returns two q-values, one for each
        Q neural network

        Args:
            state (Tensor): tensor holding current state
            action_T (Tensor): tensor holding action_T chosen by the actor
            action_R (Tensor): tensor holding action_R chosen by the actor

        Returns:
            Tuple[Tensor, Tensor]: tuple of tensors holding q-value computed by
            each Q value function.
        """
        state_actions = torch.cat([state, action_T, action_R], 1)
        return self.Q1_model(state_actions), self.Q2_model(state_actions)

    def Q1(self, state: Tensor, action_T: Tensor, action_R: Tensor) -> Tensor:
        state_actions = torch.cat([state, action_T, action_R], 1)
        return self.Q1_model(state_actions)


class Agent:
    def __init__(
        self,
        state_dim: int,
        action_T_dim: int,
        action_R_dim: int,
        discount: float,
        tau_td3: float,
        device: str,
        action_R_low: int = -1,
        action_R_high: int = 1,
    ):
        """Generates an agent object holding an actor and a critic

        Args:
            state_dim (int): Dimension of the state (num_bits)
            action_T_dim (int): Dimension of the action_T (num_reactions)
            action_R_dim (int): Dimension of the action_R (num_descriptors)
            discount (float): discount of future rewards (gamma)
            tau_td3 (float): update rate for target networks
            device (str): cuda or cpu
            action_R_low (int, optional): Lowest possible value of action_R. Defaults to -1.
            action_R_high (int, optional): Highest possible value of action_R. Defaults to 1.
        """
        self.device = device

        # Creation of actors
        self.actor = Actor(state_dim, action_T_dim, action_R_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_T_dim, action_R_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Creation of critic
        self.critic = Critic(state_dim, action_T_dim, action_R_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_T_dim, action_R_dim).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Definition of optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=3e-4)

        # Keep track of other parameters
        self.state_dim = state_dim
        self.action_T_dim = action_T_dim
        self.action_R_dim = action_R_dim
        self.action_R_low = action_R_low
        self.action_R_high = action_R_high

        self.discount = discount
        self.tau_td3 = tau_td3

    def select_action(
        self, state: Tensor, T_mask: Tensor, gumbel_tau: float, sd_noise: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given current state performs a forward pass of the actor thereby computing
        action_T and action_R. The latter gets noise added and clipped as in TD3 algorithm.

        Args:
            state (Tensor): tensor representing current state (num_bits)
            T_mask (Tensor): one-hot tensor accepting only reactions which can be used
                            with state as first reactant
            gumbel_tau (float): Temperature parameter for gumbel softmax function controlling
                            the degree of exploration / exploitation.
            sd_noise (float): SD of the noise added to the action

        Returns:
            Tuple[np.ndarray, np.ndarray]: action_T and action_R as numpy arrays
        """
        # Reshape state
        state = state.reshape(1, -1)

        # Compute actions
        actions: Tuple[Tensor, Tensor] = self.actor(state, T_mask, gumbel_tau)
        action_R: np.ndarray = actions[0].cpu().numpy().flatten()
        action_T: np.ndarray = actions[1].cpu().numpy().flatten()

        # Transform action_R by adding clipped noise and clipping as in TD3 algorithm
        action_R += np.random.normal(0, sd_noise, size=action_R.shape[0])
        action_R = action_R.clip(self.action_R_low, self.action_R_high)

        return action_T, action_R

    def train_minibatch(
        self,
        minibatch: MinibatchSample,
        update_targets: bool,
        gumbel_tau: float,
        td3_tau: float,
    ):

        # Get information from minibatch
        states = torch.Tensor(minibatch[0])
        actions_T = torch.Tensor(minibatch[1])
        # actions_R = minibatch[2]
        next_states = torch.Tensor(minibatch[3])
        t_masks = torch.Tensor(minibatch[4])
        rewards = torch.Tensor(minibatch[5])
        dones = torch.Tensor(minibatch[6])

        # Select action according to policy
        next_actions_T: Tensor
        next_actions_R: Tensor
        next_actions_T, next_actions_R, _ = self.actor_target(
            actions_T, t_masks, gumbel_tau
        )

        # Compute the target Q value
        target_Q1: Tensor
        target_Q2: Tensor
        target_Q1, target_Q2 = self.critic_target(
            states, next_actions_T, next_actions_R
        )

        target_Q: Tensor
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + ((1 - dones) * self.discount * target_Q).detach()

        # Get current Q estimates
        current_Q1: Tensor
        current_Q2: Tensor
        current_Q1, current_Q2 = self.critic(
            next_states, next_actions_T, next_actions_R
        )

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if update_targets:

            # Compute actor loss
            next_actions_T, next_actions_R, next_T_raw = self.actor(
                next_states, t_masks, gumbel_tau
            )
            f_loss = F.cross_entropy(next_T_raw, next_actions_T)

            actor_loss = -self.critic.Q1(states, next_actions_T, next_actions_R).mean()
            actor_loss += f_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    td3_tau * param.data + (1 - td3_tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    td3_tau * param.data + (1 - td3_tau) * target_param.data
                )

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )
