from torch import nn, Tensor
from typing import Tuple
import torch


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
