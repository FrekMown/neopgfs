from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from rdkit.Chem.rdChemReactions import ChemicalReaction
from typing import Tuple
import multiprocessing as mp


def inverse_reaction(rxn: ChemicalReaction) -> ChemicalReaction:
    rxn_double = ChemicalReaction()
    reactants = list(rxn.GetReactants())
    rxn_double.AddReactantTemplate(reactants[1])
    rxn_double.AddReactantTemplate(reactants[0])
    rxn_double.AddProductTemplate(list(rxn.GetProducts())[0])
    return rxn_double


def count_reactants_rxn(rxn: ChemicalReaction) -> Tuple[int, int]:
    assert rxn.GetNumReactantTemplates() == 2, "rxn must be bimolecular"
    sum_r0 = 0
    sum_r1 = 0
    for mol in reactants:
        sum_r0 += mol.HasSubstructMatch(rxn.GetReactants()[0])
        sum_r1 += mol.HasSubstructMatch(rxn.GetReactants()[1])
    return sum_r0, sum_r1


# Create relations r0/r1 rxns
def compute_reactants_for_reaction(rxn_idx):
    rxn = new_rxns[rxn_idx]
    r0 = []
    r1 = []

    for reactant_idx, mol in enumerate(reactants):
        if mol.HasSubstructMatch(rxn.GetReactants()[0]):
            r0.append(reactant_idx)
        if rxn.GetNumReactantTemplates() > 1 and mol.HasSubstructMatch(
            rxn.GetReactants()[1]
        ):
            r1.append(reactant_idx)

    return rxn_idx, r0, r1


reactions_df = pd.read_csv(
    "/home/alfredo/work/rl/neopgfs/neopgfs/data/rxn_set.txt",
    sep="|",
    header=None,
    index_col=0,
)
reactions_df.columns = ["smarts", "desc"]

with open(
    "/home/alfredo/work/rl/neopgfs/neopgfs/data/enamine_building_blocks.csv"
) as f:
    reactants_smiles = [line.strip() for line in f.readlines()[1:]]
reactants = [AllChem.MolFromSmiles(smiles) for smiles in reactants_smiles]

rxns = {
    rxn_idx: AllChem.ReactionFromSmarts(reactions_df.loc[rxn_idx, "smarts"])
    for rxn_idx in reactions_df.index
}
print(f"Number of original reactions={len(rxns)}")

to_delete = []
to_inverse = []
for rxn_idx, rxn in rxns.items():
    if rxn.GetNumReactantTemplates() == 2:
        n_r0, n_r1 = count_reactants_rxn(rxn)
        if n_r1 < 50:
            to_delete.append(rxn_idx)
        if n_r0 >= 50:
            to_inverse.append(rxn_idx)

print(f"Reactions to delete={len(to_delete)}, reactions to inverse={len(to_inverse)}")

new_reactions_df = reactions_df.drop(to_delete, axis=0)
for rxn_idx in to_inverse:
    inv_rxn = inverse_reaction(rxns[rxn_idx])
    new_reactions_df.loc[rxn_idx + " - inverse", "smarts"] = AllChem.ReactionToSmarts(
        inv_rxn
    )

print(f"Final number of reactions={len(new_reactions_df)}")

new_reactions_df.to_csv("neopgfs/data/rxn_set_v2.txt", sep="|")

print("Computing relations reactions / reactants")

## Compute arrays of relations
n_workers = mp.cpu_count() - 1
new_rxns = [AllChem.ReactionFromSmarts(smarts) for smarts in new_reactions_df.smarts]

with mp.Pool(n_workers) as pool:
    res = pool.map(compute_reactants_for_reaction, range(len(new_rxns)))

out_r0 = np.full((len(reactants), len(new_rxns)), False)
out_r1 = np.full((len(reactants), len(new_rxns)), False)

for rxn_idx, r0, r1 in res:
    for reactant_idx in r0:
        out_r0[reactant_idx, rxn_idx] = True
    for reactant_idx in r1:
        out_r1[reactant_idx, rxn_idx] = True

np.save("neopgfs/data/rel_reactant_reactions_r0.npy", out_r0)
np.save("neopgfs/data/rel_reactant_reactions_r1.npy", out_r1)

print("done")
