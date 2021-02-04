import numpy as np
from typing import Tuple, List


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, action_T, action_R, next_state, next_T_mask, reward, done)

BufferSample = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool
]
MinibatchSample = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]


class ReplayBuffer(object):
    """Buffer to store tuples of experience replay
       Saves data as (state, action_T, action_R, next_state, next_T_mask, reward, done)
    """

    storage: List[BufferSample]

    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data: BufferSample) -> None:
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple holding

        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample_minibatch(self, batch_size: int) -> MinibatchSample:
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        indices = np.random.randint(low=0, high=len(self.storage), size=batch_size)
        states, as_T, as_R, next_states, masks_t, rewards, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in indices:
            buffer_item = self.storage[i]
            states.append(buffer_item[0])
            as_T.append(buffer_item[1])
            as_R.append(buffer_item[2])
            next_states.append(buffer_item[3])
            masks_t.append(buffer_item[4])
            rewards.append(buffer_item[5])
            dones.append(buffer_item[6])

        return (
            np.array(states),
            np.array(as_T),
            np.array(as_R),
            np.array(next_states),
            np.array(masks_t),
            np.array(rewards).reshape(-1, 1),
            np.array(dones).reshape(-1, 1),
        )
