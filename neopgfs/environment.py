import numpy as np
from typing import Optional, Tuple
from .chemonster import Chemonster
from numpy.random import RandomState


class Environment:

    chemonster: Chemonster
    max_steps: int
    description: str
    # Current state stored as SMILES string
    current_state: str
    terminal: bool
    n_steps: int
    action_space: dict
    observation_space: dict
    rand_generator: RandomState

    def __init__(
        self, chemonster: Chemonster, rand_generator: RandomState, max_steps: int = 10
    ):
        self.chemonster = chemonster
        self.max_steps = max_steps
        self.description = "Chemical Environment for neoPGFS"
        self.rand_generator = rand_generator

        # Define observation space
        self.observation_space = {
            "shape": 1024,
        }

        # Define action space
        self.action_space = {
            "shape": 35,
        }

    def reset(self) -> np.ndarray:
        """Resets environment and returns a starting state

        Returns:
            np.ndarray: vector representation of current state (molecule)
        """
        self.current_state = self.chemonster.get_random_initial_molecule()
        self.n_steps = 0
        self.terminal = False
        return self.chemonster.vectorize_smiles(self.current_state, "efcp")

    def step(
        self, action_T: np.ndarray, action_R: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """Performs one training step on current state

        Args:
            action_T (int): Index of reaction template to be applied on current state
            action_R (Optional[np.ndarray], optional): If bimolecular reaction, a vector corresponding approximately
                      to a reaction must be provided. Actual reactant is computed with k nearest neighbours. Defaults to None.

        Returns:
            Tuple[np.ndarray, float, bool, np.ndarray]: returns tuple (next_state, reward, terminal, t_mask for next_state)
        """
        rxn_idx = np.where(action_T == 1)[0][0]
        self.n_steps += 1
        self.current_state, reward = self.chemonster.environment_step_pipeline(
            self.current_state, rxn_idx, action_R
        )

        t_mask = self.chemonster.compute_t_mask(self.current_state)

        # Determine if it is a terminal state
        too_many_steps = self.n_steps >= self.max_steps
        no_more_reactions = t_mask.sum() == 0

        terminal = too_many_steps or no_more_reactions

        current_state_vect = self.chemonster.vectorize_smiles(
            self.current_state, "efcp"
        )

        return current_state_vect, reward, terminal, t_mask

    def generate_random_action(self) -> Tuple[np.ndarray, np.ndarray]:

        self.n_steps += 1

        # Random selection of reaction template given current state
        t_mask = self.chemonster.compute_t_mask(self.current_state)
        allowed_reactions = np.where(t_mask > 0)[0]
        rxn_idx = self.rand_generator.choice(allowed_reactions)
        action_T = np.zeros((len(self.chemonster.reactions)))
        action_T[rxn_idx] = 1

        # Random selection of second reactant
        allowed_reactants = np.where(self.chemonster.rel_r1_rxns[:, rxn_idx] > 0)[0]
        reactant_idx = None
        if allowed_reactants.shape[0] > 0:
            reactant_idx = self.rand_generator.choice(allowed_reactants)

        return action_T, reactant_idx
