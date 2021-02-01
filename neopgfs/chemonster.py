import numpy as np
from sklearn.neighbors import NearestNeighbors
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdchem import Mol
from typing import Optional, List, Tuple
import joblib
import os
from time import time

# Paths to input
REACTANTS_PATH = "data/enamine_building_blocks.csv"
REACTIONS_PATH = "data/rxn_set.txt"
RLV2_DESCRIPTORS_PATH = "data/descriptors_rlv2.txt"
QSAR_DESCRIPTORS_PATH = "data/descriptors_qsar_models.txt"
REACTANTS_SPACE_PATH = "data/rlv2_space_reactants.npy"
REL_R0_REACTIONS = "data/rel_reactant_reactions_r0.npy"
REL_R1_REACTIONS = "data/rel_reactant_reactions_r1.npy"


class Chemonster:

    # Definition of attributes and types
    description: str
    reactants: np.ndarray
    reactions: np.ndarray
    reactants_rlv2_space: np.ndarray
    knn_models: List[NearestNeighbors]
    qsar_descriptors: List[str]
    rand_generator: np.random.RandomState
    rel_r0_rxns: np.ndarray
    rel_r1_rxns: np.ndarray

    def __init__(
        self, objective="hiv_ccr5", k=1, compute_action_space=False, seed=42,
    ):
        # Verify objective is well defined
        possible_objectives = ["hiv_ccr5", "hiv_int", "hiv_rt"]
        assert (
            objective in possible_objectives
        ), f"Objective must be one of {', '.join(possible_objectives)}"

        # Initialize random generator
        self.rand_generator = np.random.RandomState(seed)

        filedir = os.path.dirname(__file__)

        # Load reactants
        with open(os.path.join(filedir, REACTANTS_PATH)) as f:
            self.reactants = np.array([line.strip() for line in f][1:])

        # Load reactions
        with open(os.path.join(filedir, REACTIONS_PATH)) as f:
            self.reactions = np.array([line.split("|")[1] for line in f])

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

        # Add inverse reaction for bimolecular reactions
        def inverse_reaction(rxn: ChemicalReaction) -> ChemicalReaction:
            rxn_double = ChemicalReaction()
            reactants = list(rxn.GetReactants())
            rxn_double.AddReactantTemplate(reactants[1])
            rxn_double.AddReactantTemplate(reactants[0])
            rxn_double.AddProductTemplate(list(rxn.GetProducts())[0])
            return rxn_double

        for rxn_smarts in self.reactions.copy():
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            if rxn.GetNumReactantTemplates() == 2:

                rxn_inv_smarts = AllChem.ReactionToSmarts(inverse_reaction(rxn))
                self.reactions = np.append(self.reactions, [rxn_inv_smarts])
        # print(f"new reactions.shape={self.reactions.shape}")

        # Arrays holding reactants available for each reaction
        # Created with script "rel_reactant_reactions.py"
        # Dimensions: (n_reactants, n_reactions)
        self.rel_r0_rxns = np.load(os.path.join(filedir, REL_R0_REACTIONS))
        self.rel_r1_rxns = np.load(os.path.join(filedir, REL_R1_REACTIONS))
        exp_shape_rels = (len(self.reactants), len(self.reactions))
        assert (
            self.rel_r0_rxns.shape == exp_shape_rels
        ), f"Expected dimension for rel_r0_rxns is {exp_shape_rels}, got {self.rel_r0_rxns.shape}"
        assert (
            self.rel_r1_rxns.shape == exp_shape_rels
        ), f"Expected dimension for rel_r1_rxns is {exp_shape_rels}, got {self.rel_r1_rxns.shape}"

        # Identify bimolecular reactions with less than 50 second reactants
        nb_r1_per_rxn = self.rel_r1_rxns.sum(axis=0)
        reactions_less_50 = np.where(nb_r1_per_rxn < 50)[0]
        unimolecular = []  # unimolecular reactions excluded
        for arr_idx, rxn_idx in enumerate(reactions_less_50):
            rxn = self.get_reaction(rxn_idx)
            if rxn.GetNumReactantTemplates() == 1:
                unimolecular.append(arr_idx)

        # print(reactions_less_50)
        # print(unimolecular)
        reactions_less_50 = np.delete(reactions_less_50, unimolecular, 0)

        # Clear these reactions from arrays
        self.reactions = np.delete(self.reactions, reactions_less_50, 0)
        self.rel_r0_rxns = np.delete(self.rel_r0_rxns, reactions_less_50, 1)
        self.rel_r1_rxns = np.delete(self.rel_r1_rxns, reactions_less_50, 1)
        print(
            f"Final number of reactions={len(self.reactions)}, shape of relations={self.rel_r0_rxns.shape} / {self.rel_r1_rxns.shape}"
        )

        # Initialize KNN classifiers, indexed by reaction idx
        t0 = time()
        self.knn_models = []
        for reaction_idx in range(len(self.reactions)):
            reactants_reaction = np.nonzero(self.rel_r1_rxns[:, reaction_idx])[0]
            if len(reactants_reaction) > 0:
                knn_classifier = NearestNeighbors(
                    n_neighbors=np.min([k, len(reactants_reaction)])
                )
                knn_classifier.fit(self.reactants_rlv2_space[reactants_reaction])
                self.knn_models.append(knn_classifier)
            else:
                self.knn_models.append(None)

        print(
            f"{len(self.knn_models)} kNN models created and trained in {time()-t0:.2f} seconds"
        )

        # Load descriptor names for QSAR and RLV2
        with open(os.path.join(filedir, QSAR_DESCRIPTORS_PATH)) as f:
            self.qsar_descriptors = [line.strip() for line in f.readlines()]
        with open(os.path.join(filedir, RLV2_DESCRIPTORS_PATH)) as f:
            self.rlv2_descriptors = [line.strip() for line in f.readlines()]

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

    def get_reaction(self, reaction_idx: int) -> ChemicalReaction:
        """Returns ChemicalReaction at index reaction_idx

        Args:
            reaction_idx (int): Index of the reaction in original array

        Returns:
            ChemicalReaction: Object representing the given reaction
        """
        return AllChem.ReactionFromSmarts(self.reactions[reaction_idx])

    def get_reactant(self, reactant_idx: int) -> Mol:
        """Returns Mol element at index reactant_idx in the original array

        Args:
            reactant_idx (int): Index of the reactant

        Returns:
            Mol: Object representing the given reactant
        """
        return AllChem.MolFromSmiles(self.reactants[reactant_idx])

    def get_k_neighbors(self, action_R: np.ndarray, action_T: int) -> List[int]:
        """Returns k neighbors to action in RLV2 reactants space.
        Parameter k is defined at initialization of the environment.
        Takes into account only molecules that might be used for a given template

        Args:
            action_R (np.ndarray): dimensions (1 x n_RLV2_descriptors)
            action_T (int): index of the reaction to be taken into account

        Returns:
            List[int]: holds indices of k_nearest_neighbors (reactants) considering only
                       valid reactants for reaction action_T
        """
        if len(action_R.shape) == 1:
            action_R = action_R.reshape(1, -1)

        # Indices with respect to set given to self.knn_models[action_T]
        k_neighbors_rxn_ref = self.knn_models[action_T].kneighbors(
            action_R, return_distance=False
        )[0]
        # Transform indices to original scale
        return np.where(self.rel_r1_rxns[:, action_T])[0][k_neighbors_rxn_ref].tolist()

    # TO BE UPDATED
    def compute_rlv2_space_reactants(self) -> np.ndarray:
        """Computes RLV2 descriptors for all SMILES in self.reactants_df
        TO BE UPDATED

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
            mol = AllChem.MolFromSmiles(smiles)
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

        reactants = [AllChem.MolFromSmiles(r0_smiles)]

        if r1_index is not None:
            reactants.append(AllChem.MolFromSmiles(self.reactants[r1_index]))

        product = rxn.RunReactants(reactants)[0][0]

        return AllChem.MolToSmiles(product)

    def vectorize_smiles(
        self,
        smiles: str,
        method: str,
        efcp_radius: Optional[int] = None,
        efcp_length: Optional[int] = None,
    ) -> np.ndarray:
        mol = AllChem.MolFromSmiles(smiles)

        allowed_methods = ["qsar", "rlv2", "efcp"]

        assert (
            method in allowed_methods
        ), f"Method {method} not allowed. Valid ones are: {', '.join(allowed_methods)}"

        res = []

        if method == "qsar":
            res = [
                self.descriptor_functions[desc](mol) for desc in self.qsar_descriptors
            ]

        elif method == "rlv2":
            res = [
                self.descriptor_functions[desc](mol) for desc in self.rlv2_descriptors
            ]

        elif method == "efcp":  # efcp
            efcp_radius = efcp_radius or 2
            efcp_length = efcp_length or 1024
            res = AllChem.GetMorganFingerprintAsBitVect(mol, efcp_radius, efcp_length)

        return np.array(res)

    def scoring_function(self, smiles: str) -> float:
        """Calculates QSAR score using objective defined at creation of this object

        Args:
            smiles (str): SMILES of the molecule to be evaluated

        Returns:
            float: Score for the input molecule
        """
        # Calculate QSAR features for input molecule
        features = self.vectorize_smiles(smiles, "qsar").reshape(1, -1)

        predictions: List[float] = []
        for (scaler, selector, estimator) in self.models_objects:
            scaled_features = scaler.transform(features)
            selected_features = scaled_features[:, selector]
            predictions.append(estimator.predict(selected_features))

        return np.array(predictions).mean()

    def get_random_initial_molecule(self) -> str:
        """Returns a random molecule, making sure that there are reactions that
        can take it into account as R0 (reactant in first position)

        Returns:
            str: SMILES of the random molecule found
        """
        sum_reactions_r0 = self.rel_r0_rxns.sum(axis=1)
        possible_molecules_idx = np.nonzero(sum_reactions_r0)[0]
        random_molecule_idx = self.rand_generator.choice(possible_molecules_idx)
        return self.reactants[random_molecule_idx]

    def compute_t_mask(self, r0_smiles: str) -> np.ndarray:
        """Generates T-mask for a given reactant. This array holds True for reactions
        for which r0 can be taken as first reactant

        Args:
            r0_smiles (str): Smiles of reactant r0

        Returns:
            np.ndarray: dim (n_reactions, ) holding True for allowed reactions
        """
        r0 = AllChem.MolFromSmiles(r0_smiles)
        t_mask = np.full(len(self.reactions), False)

        for rxn_idx, rxn_smarts in enumerate(self.reactions):
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            if r0.HasSubstructMatch(rxn.GetReactants()[0]):
                t_mask[rxn_idx] = True
        return t_mask

    def environment_step_pipeline(
        self, current_state: str, action_T: int, action_R: Optional[np.ndarray] = None
    ) -> Tuple[str, float]:
        """Given current state, a template reaction and optionally a second reactant, this
        function computes product of the reaction as well as its score (reward)

        Args:
            current_state (str): SMILES of the current molecule (first reactant, r0)
            action_T (int): Index of the reaction to be used to compute product
            action_R (Optional[np.ndarray], optional): If reaction is bimolecular, the index of the 
            second reactant (r1) must be provided. Defaults to None.

        Returns:
            Tuple[str, float]: Returns a tuple containing smiles of product (next state), as well as
                the reward for this specific product.

        """
        # Grab reaction from template
        rxn = AllChem.ReactionFromSmarts(self.reactions[action_T])

        # If only one reactant in reaction, apply reaction and return product
        if rxn.GetNumReactantTemplates() == 1:
            next_state = self.reaction_predictor(action_T, current_state)
            reward = self.scoring_function(next_state)
        else:
            assert (
                action_R is not None
            ), "action_R should be provided for bimolecular reactions"

            k_reactant_idx = self.get_k_neighbors(action_R, action_T)  # dim (k, 1)
            k_products = [
                self.reaction_predictor(action_T, current_state, idx)
                for idx in k_reactant_idx
            ]
            k_scores = np.array(
                [self.scoring_function(smiles) for smiles in k_products]
            )
            next_state = k_products[k_scores.argmax()]
            reward = k_scores.max()

        return next_state, reward
