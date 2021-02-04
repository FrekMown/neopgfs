from torch import nn


def model_summary(model: nn.Module, name=None):
    state_dict = model.state_dict()
    max_name_length = max([len(layer) for layer in state_dict])
    print(f'Summary model {name or ""}')
    print("------------------------------------")
    total = 0
    for layer, pars in state_dict.items():
        sh = pars.shape
        total += sh.numel()
        print(f"{layer: <{max_name_length+5}} {str(tuple(sh)): <10} {sh.numel(): >10}")
    print("------------------------------------")
    print(f"** Total number of parameters: {total: ,}")
    print("------------------------------------")
