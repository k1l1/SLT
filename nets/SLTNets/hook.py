import torch


def slt_nn_hook(module, state_dict, *_):
    modules_dict = dict(module.named_modules())
    if 'frozen' in state_dict:
        frozen = state_dict.pop('frozen')
    else:
        frozen = []

    for key in state_dict:
        module_key = '.'.join(key.split('.')[0:-1])

        # The state dict contains an extra key 'frozen' that keeps track which layers are frozen
        if any(key.startswith(frozen_key) for frozen_key in frozen):
            requires_grad = False
        else:
            requires_grad = True

        if key.endswith('.weight'):
            modules_dict[module_key].weight = torch.nn.Parameter(torch.zeros(state_dict[key].shape), requires_grad=requires_grad)
        elif key.endswith('bias'):
            modules_dict[module_key].bias = torch.nn.Parameter(torch.zeros(state_dict[key].shape), requires_grad=requires_grad)
        elif key.endswith('running_mean'):
            modules_dict[module_key].running_mean = torch.nn.Parameter(torch.zeros(state_dict[key].shape), requires_grad=requires_grad)
        elif key.endswith('running_var'):
            modules_dict[module_key].running_var = torch.nn.Parameter(torch.zeros(state_dict[key].shape), requires_grad=requires_grad)
        else:
            raise NotImplementedError
