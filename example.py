import pandas as pd
from tqdm import tqdm

from neopgfs.pipeliner_light.pipelines import ClassicPipe
from neopgfs.pipeliner_light.smol import SMol

RLV2_FEATS = [
    "MaxEStateIndex",
    "MinEStateIndex",
    "MinAbsEStateIndex",
    "QED",
    "MolWt",
    "FpDensityMorgan1",
    "BalabanJ",
    "PEOE-VSA10",
    "PEOE-VSA11",
    "PEOE-VSA6",
    "PEOE-VSA7",
    "PEOE-VSA8",
    "PEOE-VSA9",
    "SMR-VSA7",
    "SlogP-VSA3",
    "SlogP-VSA5",
    "EState-VSA2",
    "EState-VSA3",
    "EState-VSA4",
    "EState-VSA5",
    "EState-VSA6",
    "FractionCSP3",
    "MolLogP",
    "Kappa2",
    "PEOE-VSA2",
    "SMR-VSA5",
    "SMR-VSA6",
    "EState-VSA7",
    "Chi4v",
    "SMR-VSA10",
    "SlogP-VSA4",
    "SlogP-VSA6",
    "EState-VSA8",
    "EState-VSA9",
    "VSA-EState9",
]

if __name__ == "__main__":

    predictions_list = []

    ccr5_pipe = ClassicPipe.load("neopgfs/models/hiv_ccr5")
    int_pipe = ClassicPipe.load("neopgfs/models/hiv_int")
    rt_pipe = ClassicPipe.load("neopgfs/models/hiv_rt")

    with open("neopgfs/data/ChEMBL_500_sample.txt", "r") as inp:
        smiles_list = inp.readlines()

    for sml_str in tqdm(smiles_list):
        smiles = sml_str.strip()
        smol = SMol(smiles)  # standardization
        smol.featurize(
            ccr5_pipe.features
            # [{"type": "ECFP"}]
        )  # same intital features set before per-model selection
        predicted_ccr5_pic_50 = ccr5_pipe.predict_vector(smol.features_values)
        predicted_int_pic_50 = int_pipe.predict_vector(smol.features_values)
        predicted_rt_pipe_pic_50 = rt_pipe.predict_vector(smol.features_values)
        predictions_list.append(
            [
                smiles,
                predicted_ccr5_pic_50,
                predicted_int_pic_50,
                predicted_rt_pipe_pic_50,
            ]
        )

    df = pd.DataFrame(
        predictions_list, columns=["SMILES", "CCR5_pIC50", "INT_pIC50", "RT_pIC50"]
    )
    df.to_csv("ChEMBL_500_sample_predicted_example.csv", index=False)
