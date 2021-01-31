import numpy as np
from typing import Optional, Tuple
from .chemonster import Chemonster


class Environment:

    chemonster: Chemonster
    max_steps: int
    description: str
    current_state: str
    terminal: bool
    n_steps: int

    def __init__(
        self, chemonster: Chemonster, max_steps: int = 10,
    ):
        self.chemonster = chemonster
        self.max_steps = max_steps
        self.description = "Chemical Environment for neoPGFS"

    def reset(self) -> str:
        self.current_state = self.chemonster.get_random_initial_molecule()
        self.n_steps = 0
        self.terminal = False
        return self.current_state

    def step(
        self, action_T: int, action_R: Optional[np.ndarray] = None
    ) -> Tuple[str, float, bool, np.ndarray]:
        """Performs one training step on current state

        Args:
            action_T (int): Index of reaction template to be applied on current state
            action_R (Optional[np.ndarray], optional): If bimolecular reaction, a vector corresponding approximately
                      to a reaction must be provided. Actual reactant is computed with k nearest neighbours. Defaults to None.

        Returns:
            Tuple[str, float, bool, np.ndarray]: returns tuple (next_state, reward, terminal, t_mask for next_state)
        """
        self.n_steps += 1
        self.current_state, reward = self.chemonster.environment_step_pipeline(
            self.current_state, action_T, action_R
        )

        t_mask = self.chemonster.compute_t_mask(self.current_state)

        # Determine if it is a terminal state
        too_many_steps = self.n_steps >= self.max_steps
        no_more_reactions = t_mask.sum() == 0

        terminal = too_many_steps or no_more_reactions

        return self.current_state, reward, terminal, t_mask

