import sys
import json
from colorama import Fore, Style
import argparse
import torch
import numpy as np

_g_mem = 0
_g_mem_activations = 0

_g_mem_dict = {'activations': []}
_g_max_activation = 0


def tensor_to_size_in_bytes(tensor):
    return int(torch.prod(torch.tensor(tensor.shape), 0))*4


def backward_hook(module, grad_input, grad_output):
    global _g_mem_activations, _g_mem_dict
    if module.__class__.__name__ == 'Identity':
        return
    if len(module._modules) == 0:
        for grad in grad_output:
            _g_mem_activations += tensor_to_size_in_bytes(grad)
            _g_mem_dict['activations'].append((module.__class__.__name__,
                                               tensor_to_size_in_bytes(grad)))
    else:
        _g_mem_dict['activations'].append((module.__class__.__name__, 0))


def forward_hook(module, input, output):
    global _g_max_activation
    if len(module._modules) == 0:
        _g_max_activation = max(_g_max_activation,
                                tensor_to_size_in_bytes(output))


def sizeof_state_dict(state_dict):
    size_in_bytes = 0
    for key in state_dict:
        tensor_dim = 1
        for dim in state_dict[key].shape:
            tensor_dim *= dim
        size_in_bytes += 4 * tensor_dim  # Conversion to bytes
    return size_in_bytes


def training_mem(Model, kwargs, input_shape, sd=None):
    global _g_mem, _g_mem_activations, _g_mem_dict, _g_max_activation
    _g_mem = 0
    _g_mem_activations = 0
    _g_mem_dict = {'activations': []}
    _g_max_activation = 0

    net = Model(**kwargs)

    if sd is not None:
        net.load_state_dict(sd)

    for module in net.modules():
        module.register_backward_hook(backward_hook)
        module.register_forward_hook(forward_hook)
    input = torch.rand(input_shape)

    out = net(input)
    out = out.sum()
    out.backward()

    size_of_loaded_parameters = sizeof_state_dict(net.state_dict())
    size_of_gradients = sizeof_state_dict({key: param for (key, param) in net.named_parameters() if param.requires_grad})

    mem_total = _g_mem_activations + size_of_loaded_parameters + size_of_gradients

    if mem_total < _g_max_activation:
        mem_total = _g_max_activation

    return round(mem_total/(10**9), 4)


def generate_configs(args):

    if args.heterogeneous:
        assert args.subset_factor < args.subset_factor_strong, 'Invalid Configuration'

    if args.model in ['ResNet20', 'ResNet44', 'ResNet56']:
        assert args.subset_factor in [0.125, 0.25, 0.5], 'Unsuported Subsetfactor'
        if args.heterogeneous:
            assert args.subset_factor_strong in [0.25, 0.5, 1.0], 'Unsuported Subsetfactor'

    if args.model in ['DenseNet']:
        assert args.subset_factor in [0.33, 0.66], 'Unsuported Subsetfactor'
        if args.heterogeneous:
            assert args.subset_factor_strong in [0.33, 0.66, 1.0], 'Unsuported Subsetfactor'

    if args.model in ['ResNet20', 'DenseNet']:
        input_shape = (32, 3, 32, 32)
    else:
        input_shape = (32, 3, 64, 64)

    if 'ResNet' in args.model:
        from utils.slt_submodel import extract_submodel_resnet_structure as extract_submodel
        from utils.slt_submodel import resnet_step_count as step_count
    else:
        from utils.slt_submodel import extract_submodel_densenet_structure as extract_submodel
        from utils.slt_submodel import densenet_step_count as step_count

    if args.model == 'ResNet20':
        from nets.SLTNets.resnet_cifar import ResNet20 as SLTNet
        from nets.SubsetNets.resnet_cifar import ResNet20 as SubsetNet
    elif args.model == 'ResNet44':
        from nets.SLTNets.resnet_cifar import ResNet44 as SLTNet
        from nets.SubsetNets.resnet_cifar import ResNet44 as SubsetNet
    elif args.model == 'ResNet56':
        from nets.SLTNets.resnet_cifar import ResNet56 as SLTNet
        from nets.SubsetNets.resnet_cifar import ResNet56 as SubsetNet
    elif args.model == 'DenseNet':
        from nets.SLTNets.densenet_cifar import DenseNet40 as SLTNet
        from nets.SubsetNets.densenet_cifar import DenseNet40 as SubsetNet

    print('Subset Evaluation')
    sd = SubsetNet().state_dict()
    memory_limit = training_mem(SubsetNet, {'subset_factor': args.subset_factor}, input_shape=input_shape)
    print('Subset ratio: ', args.subset_factor, 'max memory consumption: ', memory_limit)

    if args.heterogeneous:
        print('Subset Evaluation (heterongeous) of strong device')
        sd = SubsetNet().state_dict()
        memory_limit_heterogeneous = training_mem(SubsetNet, {'subset_factor': args.subset_factor_strong}, input_shape=input_shape)
        print('Subset ratio (strong device): ', args.subset_factor_strong, 'max memory consumption: ', memory_limit_heterogeneous)

    # SLT evaluation
    sd_standard = SLTNet().state_dict()
    sd = SLTNet().state_dict()
    num_total_steps = step_count(sd_standard)
    print('total number of SLT steps:', num_total_steps)
    steps_normalized = [np.ceil(100*i/num_total_steps)/100.0 for i in range(0, num_total_steps)]

    results = []
    s = 1.0  # Starting value
    for idx, step in reversed(list(enumerate(steps_normalized))):
        found = False
        while not found:
            sd = SLTNet().state_dict()
            sd_reduced, indices, freeze_dict = extract_submodel(round(s, 4),
                                                    step, sd, training_depth=1)

            for key in sd_reduced:
                sd_reduced[key].data.uniform_(-1e-5, 1e-5)

            sd_reduced.update({'frozen': freeze_dict})
            memory = training_mem(SLTNet, {}, input_shape=input_shape, sd=sd_reduced)

            if memory > memory_limit:
                s = s - 0.001
                if s <= 0.0:
                    raise ValueError
            else:
                print(str(idx).ljust(2), str(round(s, 4)).ljust(5),
                    str(step).ljust(4), Fore.GREEN, memory, Style.RESET_ALL)
                results.append([round(s, 4), step])
                found = True

    json_result = []
    for s, step in list(reversed(results))[1:]:
        json_result.append([s, step])
        # Only store results until model can be fully trained
        if s == 1.0:
            break

    if not args.heterogeneous:
        file_path = f'memory/configs/config__{args.model.lower()}.json'
        try:
            with open(file_path, 'r') as fd:
                data = json.load(fd)
        except FileNotFoundError:
            data = {}
        try:
            data[f'{args.subset_factor}']['values'] = json_result
        except KeyError:
            data.update({f'{args.subset_factor}': {'values': json_result}})
        with open(file_path, 'w') as fd:
            json.dump(data, fd, indent=4)
    else:
        print('Evaluating training depth of strong device (hetoerogenous case)')
        training_depth = num_total_steps
        training_depth_list = []
        for idx, (s, step) in enumerate(results):
            found = False
            while not found:
                sd = SLTNet().state_dict()
                sd_reduced, indices, freeze_dict = extract_submodel(s, step,
                                    sd, training_depth=training_depth)

                for key in sd_reduced:
                    sd_reduced[key].data.uniform_(-1e-5, 1e-5)

                sd_reduced.update({'frozen': freeze_dict})
                memory = training_mem(SLTNet, {}, input_shape=input_shape, sd=sd_reduced)

                if memory > memory_limit_heterogeneous:
                    training_depth = training_depth - 1

                    # For consistency, training depth can not be deeper
                    # than available layers
                    training_depth = min(training_depth, num_total_steps - 1 - idx)
                    if training_depth < 0:
                        raise ValueError
                else:
                    # For consistency, training depth can not be deeper than
                    # available layers
                    training_depth = min(training_depth, num_total_steps - 1 - idx)
                    print(str(num_total_steps - 1 - idx).ljust(2), str(s).ljust(5), str(step).ljust(4), 'training depth', training_depth, Fore.GREEN, memory, Style.RESET_ALL)
                    found = True
                    training_depth_list.append(training_depth)

        training_depth_data = []
        for depth, (s, _) in zip(list(reversed(training_depth_list))[1:], list(reversed(results))[1:]):
            training_depth_data.append(depth)
            # Only store results until model can be fully trained
            if s == 1.0:
                break

        file_path = f'memory/configs/config__{args.model.lower()}.json'
        try:
            with open(file_path, 'r') as fd:
                data = json.load(fd)
        except FileNotFoundError:
            data = {}
        try:
            data[f"{args.subset_factor}"][args.subset_factor_strong]['training_depth'] = training_depth_data
        except KeyError:
            try:
                data[f"{args.subset_factor}"].update({f"{args.subset_factor_strong}": {'training_depth': training_depth_data}})
            except KeyError:
                data.update({f"{args.subset_factor}": {'values': json_result}})
                data[f'{args.subset_factor}'].update({f'{args.subset_factor_strong}': {'training_depth': training_depth_data}})

        with open(file_path, 'w') as fd:
            json.dump(data, fd, indent=4)


if __name__ == '__main__':

    sys.path[0] = sys.path[0][:-len('memory')]
    MODELS = ['ResNet20', 'DenseNet', 'ResNet44', 'ResNet56']

    parser = argparse.ArgumentParser(description='SLT configuration generator. Constraints are evaluated relative to a downscaled baseline. The final configurations are stored in memory/configs/config__MODEL.json.')
    parser.add_argument('--model', choices=MODELS, type=str, default=MODELS[2], help=f'NN model to choose from ({MODELS}).')
    parser.add_argument('--subset_factor', type=float, default=0.125, help='Subset factor to select. For ResNet structures, 0.125, 0.25, 0.5 is available, for DenseNet 0.33 and 0.66.')
    parser.add_argument('--heterogeneous', action='store_true', default=False, help='If set to True, a second subset factor of a strong device is evaluated, specifically, the training depth of the stronger device.')
    parser.add_argument('--subset_factor_strong', type=float, default=0.5, help='Sets the subset factor of the strong device.Choices are similar to weaker device. The subsetfactor of the strong device must be greater than the one of the weak device.')

    args = parser.parse_args()

    generate_configs(args)
