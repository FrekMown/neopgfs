import numpy as np
from .environment import Environment
from .agent import Agent
from .replay_buffer import ReplayBuffer


class Runner:
    """Carries out the environment steps and adds experiences to memory"""

    def __init__(self, env: Environment, agent: Agent, replay_buffer: ReplayBuffer):

        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done: bool = False

    def next_step(self, episode_timesteps, replay_buffer, noise=0.1):

        action = self.agent.select_action(np.array(self.obs), noise)

        # Perform action
        new_obs, reward, done, _ = self.env.step(action)
        done_bool = 0 if episode_timesteps + 1 == 200 else float(done)

        # Store data in replay buffer
        replay_buffer.add((self.obs, new_obs, action, reward, done_bool))

        self.obs = new_obs

        if done:
            self.obs = self.env.reset()
            done = False

            return reward, True

        return reward, done
