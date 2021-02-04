import unittest
import sys
from os.path import dirname, abspath, join
import numpy as np

import torch

path = abspath(join(dirname(__file__), "..", ".."))
sys.path.append(path)

from neopgfs.chemonster import Chemonster  # noqa
from neopgfs.agent import Agent
from neopgfs.actor import Actor
from neopgfs.critic import Critic


class TestAgent(unittest.TestCase):
    state_dim = 1024
    action_T_dim = 99
    action_R_dim = 35
    batch_size = 32
    device = "cpu"

    state = torch.rand((batch_size, state_dim)).to(device)
    action_T = torch.rand((batch_size, action_T_dim)).to(device)
    T_mask = torch.rand((batch_size, action_T_dim)).to(device)
    action_R = torch.rand((batch_size, action_R_dim)).to(device)
    gumbel_tau = 0.1
    sd_noise = 0.1

    agent = Agent(
        state_dim=state_dim,
        action_T_dim=action_T_dim,
        action_R_dim=action_R_dim,
        discount=0.99,
        tau_td3=0.1,
        device=device,
    )

    critic = Critic(
        state_dim=state_dim,
        action_T_dim=action_T_dim,
        action_R_dim=action_R_dim,
        device=device,
    )

    actor = Actor(
        state_dim=state_dim,
        action_T_dim=action_T_dim,
        action_R_dim=action_R_dim,
        device=device,
    )

    def test_actor(self):
        """Verifies correctness of shapes
        """
        output_f = self.actor.f(self.state)
        output_pi = self.actor.pi(torch.cat([self.state, self.action_T], 1))
        output_actor = self.actor(self.state, self.T_mask, self.gumbel_tau)
        self.assertEqual(tuple(output_f.shape), (self.batch_size, self.action_T_dim))
        self.assertEqual(tuple(output_pi.shape), (self.batch_size, self.action_R_dim))
        output_T, output_R, output_T_mask = output_actor
        self.assertEqual(tuple(output_T.shape), (self.batch_size, self.action_T_dim))
        self.assertEqual(tuple(output_R.shape), (self.batch_size, self.action_R_dim))
        self.assertEqual(
            tuple(output_T_mask.shape), (self.batch_size, self.action_T_dim)
        )

    def test_critic(self):
        q_input = torch.cat([self.state, self.action_T, self.action_R], 1)
        output_Q1 = self.critic.Q1_model(q_input)
        output_Q2 = self.critic.Q2_model(q_input)
        output_critic_Q1, output_critic_Q2 = self.critic(
            self.state, self.action_T, self.action_R
        )
        self.assertEqual(tuple(output_Q1.shape), (self.batch_size, 1))
        self.assertEqual(tuple(output_Q2.shape), (self.batch_size, 1))
        self.assertEqual(tuple(output_critic_Q1.shape), (self.batch_size, 1))
        self.assertEqual(tuple(output_critic_Q2.shape), (self.batch_size, 1))

    def test_agent(self):
        output_agent = self.agent.select_action(
            self.state[0],
            self.T_mask[0],
            gumbel_tau=self.gumbel_tau,
            sd_noise=self.sd_noise,
        )
        agent_T, agent_R, agent_T_mask = output_agent

        self.assertEqual(tuple(agent_T.shape), (self.action_T_dim,))
        self.assertEqual(tuple(agent_R.shape), (self.action_R_dim,))
        self.assertEqual(tuple(agent_T_mask.shape), (self.action_T_dim,))


class TestChemonster(unittest.TestCase):
    chemonster = Chemonster(k=5, objective="hiv_int", seed=42)

    def test_k_neighbors(self):
        action = np.load("action_test.npy")
        reaction_index = 25
        result = [2552, 2641, 5373, 3178, 3232]
        self.assertEqual(
            self.chemonster.get_k_neighbors(action, reaction_index), result
        )

    def test_reaction_predictor(self):
        r0_smiles = "CC#Cc1cccc(B(O)O)c1"
        r1_idx = 1747
        rxn_idx = 96
        exp_result = "C#Cc1cn(-c2cccc(C#CC)c2)cn1"
        result = self.chemonster.reaction_predictor(rxn_idx, r0_smiles, r1_idx)
        self.assertEqual(exp_result, result)

    def test_scoring_function(self):
        test_smiles = "CCOC(=O)c1nc(Nc2ccc(OC)cc2)c2ccccc2n1"
        exp_result = 5.234701518308217
        result = self.chemonster.scoring_function(test_smiles)
        self.assertAlmostEqual(result, exp_result)

    def test_initial_random_molecule(self):
        result = self.chemonster.get_random_initial_molecule()
        idx_reactant = np.nonzero(self.chemonster.reactants == result)[0][0]
        sum_r0_reactions = self.chemonster.rel_r0_rxns.sum(axis=1)[idx_reactant]
        self.assertGreater(sum_r0_reactions, 0)

    def test_compute_t_mask(self):
        idx_molecule = 38585  # the most reactive one
        smiles_molecule = self.chemonster.reactants[idx_molecule]
        res = self.chemonster.compute_t_mask(smiles_molecule)
        exp_res = self.chemonster.rel_r0_rxns[idx_molecule]
        self.assertTrue(np.all(exp_res == res))

    def test_vectorize_smiles(self):
        data_test = np.load("test_featurize_smiles.npz")
        smiles = data_test["smiles"][0]
        res_efcp = self.chemonster.vectorize_smiles(smiles, "efcp", 2, 1024)
        res_rlv2 = self.chemonster.vectorize_smiles(smiles, "rlv2")
        res_qsar = self.chemonster.vectorize_smiles(smiles, "qsar")
        self.assertTrue(np.all(res_efcp == data_test["efcp"]))
        self.assertTrue(np.all(res_rlv2 == data_test["rlv2"]))
        self.assertTrue(np.all(res_qsar == data_test["qsar"]))

    def test_env_step_pipeline(self):
        curr_state = "CCN(CC)C(=O)c1ccc(C=O)cc1"
        rxn_idx = 31
        r1_rlv2_noisy = np.load("action_R_test_pipeline.npy")
        exp_smiles = "CCN(CC)C(=O)c1ccc(CN[C@@H](Cc2c[nH]c3ccccc23)C(=O)N[C@@H](CCCNC(=N)N)C(=O)Nc2cccc(C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(N)=O)c2)cc1"
        exp_reward = 5.735187515665501
        smiles, reward = self.chemonster.environment_step_pipeline(
            curr_state, rxn_idx, r1_rlv2_noisy
        )
        self.assertEqual(smiles, exp_smiles)
        self.assertAlmostEqual(reward, exp_reward)


if __name__ == "__main__":
    unittest.main()
