from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdChemReactions import ChemicalReaction
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
from typing import Optional, List, Tuple
import joblib

# Paths to input
REACTANTS_PATH = "data/enamine_building_blocks.csv"
REACTIONS_PATH = "data/rxn_set.txt"
RLV2_DESCRIPTORS_PATH = "data/descriptors_rlv2.txt"
QSAR_DESCRIPTORS_PATH = "data/descriptors_qsar_models.txt"
REACTANTS_SPACE_PATH = "data/rlv2_space_reactants.npy"
REL_R0_REACTIONS = "data/rel_reactant_reactions_r0.npy"
REL_R1_REACTIONS = "data/rel_reactant_reactions_r1.npy"


class Environment:

    # Definition of attributes and types
    description: str
    reactants: np.ndarray
    reactions: np.ndarray
    reactants_rlv2_space: np.ndarray
    knn_classifier: NearestNeighbors
    qsar_descriptors: List[str]

    def __init__(
        self,
        objective="hiv_ccr5",
        k=1,
        compute_action_space=False,
        seed=42,
        max_steps: int = 10,
    ):

        # Verify objective is well defined
        possible_objectives = ["hiv_ccr5", "hiv_int", "hiv_rt"]
        assert (
            objective in possible_objectives
        ), f"Objective must be one of {', '.join(possible_objectives)}"

        # Initialize random generator
        self.rand_generator = np.random.RandomState(seed)

        # Initialize current state
        self.max_steps = max_steps

        filedir = os.path.dirname(__file__)

        self.description = "Chemical Environment for neoPGFS"

        # Load reactants
        with open(os.path.join(filedir, REACTANTS_PATH)) as f:
            self.reactants = np.array([line.strip() for line in f][1:])

        # Load reactions
        with open(os.path.join(filedir, REACTIONS_PATH)) as f:
            self.reactions = np.array([line.split("|")[1] for line in f])

        # Arrays holding reactants available for each reaction
        # Created with script "rel_reactant_reactions.py"
        self.rel_r0_rxns = np.load(os.path.join(filedir, REL_R0_REACTIONS))
        self.rel_r1_rxns = np.load(os.path.join(filedir, REL_R1_REACTIONS))

        print(
            f"reactants.shape={self.reactants.shape}, reactions.shape={self.reactions.shape}"
        )

        # Compute action RLV2 space, dims (n_reactants x n_descriptors)
        if compute_action_space:
            self.reactants_rlv2_space = self.compute_rlv2_space_reactants()
            np.save(
                os.path.join(filedir, REACTANTS_SPACE_PATH), self.reactants_rlv2_space
            )
        else:
            self.reactants_rlv2_space = np.load(
                os.path.join(filedir, REACTANTS_SPACE_PATH)
            )

        # Initialize KNN classifier
        self.knn_classifier = NearestNeighbors(n_neighbors=k)
        self.knn_classifier.fit(self.reactants_rlv2_space)

        # Add inverse reaction for bimolecular reactions
        def inverse_reaction(rxn: ChemicalReaction) -> ChemicalReaction:
            rxn_double = ChemicalReaction()
            reactants = list(rxn.GetReactants())
            rxn_double.AddReactantTemplate(reactants[1])
            rxn_double.AddReactantTemplate(reactants[0])
            rxn_double.AddProductTemplate(list(rxn.GetProducts())[0])
            return rxn_double

        for rxn_idx, rxn_smarts in enumerate(self.reactions.copy()):
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            if rxn.GetNumReactantTemplates() == 2:
                rxn_inv_smarts = AllChem.ReactionToSmarts(inverse_reaction(rxn))
                self.reactions = np.append(self.reactions, [rxn_inv_smarts])
        print(f"new reactions.shape={self.reactions.shape}")

        # Load descriptors for QSAR models
        with open(os.path.join(filedir, QSAR_DESCRIPTORS_PATH)) as f:
            self.qsar_descriptors = [line.strip() for line in f.readlines()]

        # Load dictionary of descriptor functions from RDKit
        self.descriptor_functions = dict(Descriptors.descList)

        # Load data related to QSAR models for given objective
        # models_objects contain a list of tuples (scaler, selector, estimator)
        # to be applied to mol descriptors
        models_dir = os.path.join(filedir, "models", objective)
        self.models_objects = []
        for model_name in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, model_name)):
                scaler = joblib.load(os.path.join(models_dir, model_name, "scaler.sav"))
                selector = np.load(os.path.join(models_dir, model_name, "selector.npy"))
                estimator = joblib.load(
                    os.path.join(models_dir, model_name, "estimator.sav")
                )
                self.models_objects.append((scaler, selector, estimator))
        print(f"Loaded {len(self.models_objects)} QSAR models")

    def get_k_neighbors(self, action_R: np.ndarray) -> List[int]:
        """Returns k neighbors to action in RLV2 reactants space.
        Parameter k is defined at initialization of the environment

        Args:
            action_R (np.ndarray): dimensions (1 x n_RLV2_descriptors)

        Returns:
            List[int]: holds indices of k_nearest_neighbors (reactants)
        """

        return self.knn_classifier.kneighbors(action_R, return_distance=False)[
            0
        ].tolist()

    def compute_rlv2_space_reactants(self) -> np.ndarray:
        """Computes RLV2 descriptors for all SMILES in self.reactants_df

        Returns:
            np.ndarray: dimensions (n_reactants x n_descriptors) representing dimensions
                        for each reactant. List of descriptors is read from file
                        in RLV2_DESCRIPTORS_PATH.
        """
        # Load list of RLV2 descriptors
        with open(RLV2_DESCRIPTORS_PATH) as f:
            descriptors_list = [line.strip() for line in f.readlines()]

        # Initialize rlv2_space
        rlv2_space = np.zeros((len(self.reactants), len(descriptors_list)))

        for n_reactant, smiles in enumerate(self.reactants):
            if n_reactant % 50000 == 0:
                print(f"n_reactant: {n_reactant}")
            mol = Chem.MolFromSmiles(smiles)
            for n_descriptor, descriptor in enumerate(descriptors_list):
                rlv2_space[n_reactant, n_descriptor] = self.descriptor_functions[
                    descriptor
                ](mol)

        return rlv2_space

    def reaction_predictor(
        self, template_idx: int, r0_smiles: str, r1_index: Optional[int] = None,
    ) -> str:
        """Given a reaction template, a first reactant (r0) and optionally a second
        one (r1), computes the product of the reaction and returns it as SMILES.
        Note: returns only the first product

        Args:
            template_idx (int): Index of the reaction template
            r0_smiles (str): Smiles of the current reactant (state)
            r1_index (Optional[int], optional): Index of the second reactant, if any. Defaults to None.

        Returns:
            str: SMILES of the first product of the reaction applied to both reactants
        """
        rxn = AllChem.ReactionFromSmarts(self.reactions[template_idx])

        reactants = [Chem.MolFromSmiles(r0_smiles)]

        if r1_index is not None:
            reactants.append(Chem.MolFromSmiles(self.reactants[r1_index]))

        product = rxn.RunReactants(reactants)[0][0]

        return Chem.MolToSmiles(product)

    def scoring_function(self, smiles: str) -> float:

        # Calculate QSAR features for input molecule
        mol = Chem.MolFromSmiles(smiles)
        features = np.array(
            [self.descriptor_functions[desc](mol) for desc in self.qsar_descriptors]
        ).reshape(1, -1)

        predictions: List[float] = []
        for (scaler, selector, estimator) in self.models_objects:
            scaled_features = scaler.transform(features)
            selected_features = scaled_features[:, selector]
            predictions.append(estimator.predict(selected_features))

        return np.array(predictions).mean()

    def reset(self) -> str:
        self.current_state = np.random.choice(self.reactants)
        self.n_steps = 0
        self.terminal = False
        return self.current_state

    def step(
        self, action_T: int, action_R: Optional[np.ndarray] = None
    ) -> Tuple[str, float, bool]:

        self.n_steps += 1

        # Grab reaction from template
        rxn = AllChem.ReactionFromSmarts(self.reactions[action_T])

        # If only one reactant in reaction, apply reaction and return product
        if rxn.GetNumReactantTemplates() == 1:
            self.current_state = self.reaction_predictor(action_T, self.current_state)
        else:
            assert (
                action_R is not None
            ), "action_R should be provided for bimolecular reactions"

            k_reactant_idx = self.get_k_neighbors(action_R)  # dim (k, 1)
            k_products = [
                self.reaction_predictor(action_T, self.current_state, idx)
                for idx in k_reactant_idx
            ]
            k_scores = np.array(
                [self.scoring_function(smiles) for smiles in k_products]
            )

            self.current_state = k_products[k_scores.argmax()[0]]
            reward = k_scores.max()
            self.terminal = self.n_steps >= self.max_steps

        return (self.current_state, reward, self.terminal)

        # Get k products

        k_products = [self.reaction_predictor(action_T, self.current_state,)]
        self.reaction_predictor()

