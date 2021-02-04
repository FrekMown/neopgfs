from torch import nn, Tensor
import torch.nn.functional as F
import torch
from typing import Tuple


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_T_dim: int, action_R_dim: int, device: str,
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
        ).to(device)

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
        ).to(device)

    def forward(
        self, state: Tensor, T_mask: Tensor, gumbel_tau: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Makes a forward pass for the actor, returns action_R and action_T.

        Args:
            state (Tensor): tensor representing current state
            T_mask (Tensor): one-hot tensor representing the reactions that might be used with current state as first reactant
            gumbel_tau (float): temperature parameter for Gumbel Softmax function,
            controlling the degree of exploration to be performed

        Returns:
            Tuple[Tensor, Tensor, Tensor]: action_T representing the chosen reaction template, action_R representing the chosen second reactant, and T representing the raw output of the f network
        """
        # Choose action_T using f network
        T_raw: Tensor = self.f(state)

        # Transform action_T to one hot vector using Gumbel Softmax
        action_T = F.gumbel_softmax(T_raw * T_mask, gumbel_tau, hard=True)

        # Choose action_R using policy network
        action_R: Tensor = self.pi(torch.cat((state, action_T), dim=-1))

        return action_T, action_R, T_raw
