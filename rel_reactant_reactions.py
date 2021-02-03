from rdkit.Chem import AllChem
import multiprocessing as mp
import numpy as np
from time import time
from neopgfs.chemonster import Chemonster

t0 = time()

chemonster = Chemonster()
n_workers = mp.cpu_count() - 1


def compute_reactants_for_reaction(rxn_idx):
    rxn = AllChem.ReactionFromSmarts(chemonster.reactions[rxn_idx])
    r0 = []
    r1 = []

    for reactant_idx, reactant_smiles in enumerate(chemonster.reactants):
        mol = AllChem.MolFromSmiles(reactant_smiles)
        if mol.HasSubstructMatch(rxn.GetReactants()[0]):
            r0.append(reactant_idx)
        if rxn.GetNumReactantTemplates() > 1 and mol.HasSubstructMatch(
            rxn.GetReactants()[1]
        ):
            r1.append(reactant_idx)

    return rxn_idx, r0, r1


with mp.Pool(n_workers) as pool:
    res = pool.map(compute_reactants_for_reaction, range(chemonster.reactions.shape[0]))

out_r0 = np.full((len(chemonster.reactants), len(chemonster.reactions)), False)
out_r1 = np.full((len(chemonster.reactants), len(chemonster.reactions)), False)

for rxn_idx, r0, r1 in res:
    for reactant_idx in r0:
        out_r0[reactant_idx, rxn_idx] = True
    for reactant_idx in r1:
        out_r1[reactant_idx, rxn_idx] = True

np.save("neopgfs/data/rel_reactant_reactions_r0.npy", out_r0)
np.save("neopgfs/data/rel_reactant_reactions_r1.npy", out_r1)

print(f"Script finished in {time()-t0:.0f} seconds")
