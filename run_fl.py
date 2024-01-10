import json
import hashlib
import os
from datetime import datetime
import argparse
import logging
from copy import deepcopy


def dict_hash(dictionary):
    '''MD5 hash of a dictionary.'''
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def prepare(settings):

    assert settings.n_devices >= settings.n_devices_per_round, 'Cannot be more active devices than overall devices'
    assert settings.n_devices > 0
    assert settings.n_devices_per_round > 0

    # Compatability checks
    if settings.network in ['ResNet20', 'DenseNet40']:
        if settings.dataset not in ['CIFAR10', 'CIFAR100', 'FEMNIST']:
            raise ValueError(f'{settings.dataset} Incompatible combination of dataset and network')
        if any(settings.memory_constraint) not in [1.0, 0.5, 0.25, 0.125]:
            raise ValueError(settings.memory_constraint, f'{settings.network} only supports 1.0, 0.5, 0.25, or 0.125')

    if settings.network in ['ResNet44', 'ResNet56']:
        if settings.dataset not in ['TinyImageNet', 'ImageNet']:
            raise ValueError(f'{settings.dataset} Incompatible combination of dataset and network')
        if any(settings.memory_constraint) not in [1.0, 0.66, 0.33]:
            raise ValueError(settings.memory_constraint, f'{settings.network} only supports 1.0, 0.66, or 0.33')

    settings_as_dict = vars(deepcopy(settings))
    settings_as_dict.pop('n_rounds')
    settings_as_dict.pop('torch_device')
    settings_as_dict.pop('progress_bar')
    settings_as_dict.pop('plot')
    settings_as_dict.pop('data_path_prefix')
    settings_as_dict.pop('dry_run')

    if settings.data_distribution == 'iid':
        settings_as_dict.pop('distribution_dirichlet_alpha')

    run_hash = dict_hash(settings_as_dict)

    print('{' + '\n'.join('{!r}: {!r},'.format(k, v) for k, v in settings_as_dict.items()) + '}')

    # Set GPU's
    if settings.torch_device.startswith('cuda'):
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = settings.torch_device.split('cuda:')[1]
            settings.torch_device = 'cuda'
        except IndexError:
            pass

    import torch
    torch.manual_seed(settings.seed)
    from torchvision import datasets

    cnn_args = {}
    device_cnn_args = {}
    run_path = 'runs/' + settings.session_tag + '/run_'


    # Algorithm
    if settings.algorithm == 'FedAvg':
        from algorithms.fedavg import FedAvgServer as Server
    elif settings.algorithm == 'FedRolex':
        from algorithms.fedrolex import FedRolexServer as Server
    elif settings.algorithm == 'FederatedDropout':
        from algorithms.fd import FederatedDropoutServer as Server
    elif settings.algorithm == 'SLT':
        from algorithms.slt import SLTServer as Server
    elif settings.algorithm == 'HeteroFL':
        from algorithms.heterofl import HeteroFLServer as Server
    elif settings.algorithm == 'FjORD':
        from algorithms.fjord import FjordServer as Server
    else:
        raise ValueError(settings.algorithm)

    flserver = Server(run_path + run_hash)
      
    # Set Keep factor list
    if settings.algorithm == 'FedAvg':
        # Small model has to fallback to smallest model in heterogeneous case
        minimal_memoy_constraint = min(settings.memory_constraint)
        flserver.memory_constraints_list = [minimal_memoy_constraint for _ in range(settings.n_devices)]
    else:
        flserver.memory_constraints_list = [settings.memory_constraint[
            int(i*len(settings.memory_constraint)/settings.n_devices)] for i in range(settings.n_devices)]

    flserver.n_devices_per_round = settings.n_devices_per_round
    flserver.n_devices = settings.n_devices
    flserver.torch_device = settings.torch_device
    flserver.n_rounds = settings.n_rounds
    flserver.lr = settings.lr
    flserver.lr_min = float(settings.lr/settings.lr_reduction_factor)
    flserver.set_seed(settings.seed)

    flserver.progress_output = settings.progress_bar
    flserver.set_optimizer(torch.optim.SGD, {'weight_decay': settings.weight_decay, 'momentum': 0.9})

    net_eval = None

    from nets.SubsetNets.resnet_cifar import ResNet20, ResNet44
    from nets.SubsetNets.densenet_cifar import DenseNet40

    # Models
    if settings.network == 'ResNet20':
        if settings.algorithm == 'SLT':
            from nets.SLTNets.resnet_cifar import ResNet20 as SLTResNet20
            from utils.slt_submodel import extract_submodel_resnet_structure
            net = SLTResNet20
            net_eval = SLTResNet20
            flserver.nn_configs_path = 'nets/SLTNets/configs/config__resnet20.json'
            flserver.extract_fnc = extract_submodel_resnet_structure
        else:
            net = ResNet20
            net_eval = ResNet20
            if settings.algorithm == 'FederatedDropout':
                from utils.federated_dropout_submodel import extract_submodel_resnet_structure
                flserver.extract_fnc = extract_submodel_resnet_structure
            elif settings.algorithm == 'FedRolex':
                from utils.fedrolex_submodel import extract_submodel_resnet_structure
                flserver.extract_fnc = extract_submodel_resnet_structure
            elif settings.algorithm == 'HeteroFL' or settings.algorithm == 'FjORD':
                from utils.heterofl_submodel import extract_submodel_resnet_structure
                flserver.extract_fnc = extract_submodel_resnet_structure
    elif settings.network == 'ResNet44':
        from nets.SubsetNets.resnet_cifar import ResNet44 as ResNet44
        if settings.algorithm == 'SLT':
            from nets.SLTNets.resnet_cifar import ResNet44 as SLTResNet44
            from utils.slt_submodel import extract_submodel_resnet_structure
            net = SLTResNet44
            net_eval = SLTResNet44
            flserver.nn_configs_path = 'nets/SLTNets/configs/config__resnet44.json'
            flserver.extract_fnc = extract_submodel_resnet_structure
        else:
            net = ResNet44
            net_eval = ResNet44
            if settings.algorithm == 'FederatedDropout':
                from utils.federated_dropout_submodel import extract_submodel_resnet_structure
                flserver.extract_fnc = extract_submodel_resnet_structure
            elif settings.algorithm == 'FedRolex':
                from utils.fedrolex_submodel import extract_submodel_resnet_structure
                flserver.extract_fnc = extract_submodel_resnet_structure
            elif settings.algorithm == 'HeteroFL' or settings.algorithm == 'FjORD':
                from utils.heterofl_submodel import extract_submodel_resnet_structure
                flserver.extract_fnc = extract_submodel_resnet_structure
    elif settings.network == 'ResNet56':
            from nets.SubsetNets.resnet_cifar import ResNet56 as ResNet56
            if settings.algorithm == 'SLT':
                from nets.SLTNets.resnet_cifar import ResNet56 as SLTResNet56
                from utils.slt_submodel import extract_submodel_resnet_structure
                net = SLTResNet56
                net_eval = SLTResNet56
                flserver.nn_configs_path = 'nets/SLTNets/configs/config__resnet56.json'
                flserver.extract_fnc = extract_submodel_resnet_structure
            else:
                net = ResNet56
                net_eval = ResNet56
                if settings.algorithm == 'FederatedDropout':
                    from utils.federated_dropout_submodel import extract_submodel_resnet_structure
                    flserver.extract_fnc = extract_submodel_resnet_structure
                elif settings.algorithm == 'FedRolex':
                    from utils.fedrolex_submodel import extract_submodel_resnet_structure
                    flserver.extract_fnc = extract_submodel_resnet_structure
                elif settings.algorithm == 'HeteroFL' or settings.algorithm == 'FjORD':
                    from utils.heterofl_submodel import extract_submodel_resnet_structure
                    flserver.extract_fnc = extract_submodel_resnet_structure
    elif settings.network == 'DenseNet40':
        if settings.algorithm == 'SLT':
            from nets.SLTNets.densenet_cifar import DenseNet40 as SLTDenseNet40
            from utils.slt_submodel import extract_submodel_densenet_structure

            net = SLTDenseNet40
            net_eval = SLTDenseNet40
            flserver.nn_configs_path = 'nets/SLTNets/configs/config__densenet40.json'
            flserver.extract_fnc = extract_submodel_densenet_structure
        else:
            net = DenseNet40
            net_eval = DenseNet40
            if settings.algorithm == 'FederatedDropout':
                from utils.federated_dropout_submodel import extract_submodel_densenet_structure
                flserver.extract_fnc = extract_submodel_densenet_structure
            elif settings.algorithm == 'FedRolex':
                from utils.fedrolex_submodel import extract_submodel_densenet_structure
                flserver.extract_fnc = extract_submodel_densenet_structure
            elif settings.algorithm == 'HeteroFL' or settings.algorithm == 'FjORD':
                from utils.heterofl_submodel import extract_submodel_densenet_structure
                flserver.extract_fnc = extract_submodel_densenet_structure
    else: 
        raise ValueError(settings.network)

    # Data Split
    from utils.split import split_iid, split_noniid
    if settings.data_distribution == 'noniid':
        flserver.split_f = split_noniid(settings.distribution_dirichlet_alpha, run_path + run_hash,
                                                settings.plot if not settings.dry_run else False, settings.seed)
    else:
        flserver.split_f = split_iid(run_path + run_hash, settings.plot if not settings.dry_run else False, settings.seed)

    # Dataset
    if 'CIFAR' in settings.dataset:
        from utils.datasets.cifar import tf_cifar_train, tf_cifar_test
        kwargs = {'download': True}
        if settings.dataset.endswith('100'):
            flserver.set_dataset(datasets.CIFAR100, settings.pat, kwargs, tf_cifar_train, tf_cifar_test)
            cnn_args.update({'num_classes': 100})
        elif settings.dataset.endswith('10'):
            flserver.set_dataset(datasets.CIFAR10, settings.data_path_prefix, kwargs, tf_cifar_train, tf_cifar_test)
            cnn_args.update({'num_classes': 10})
    elif 'FEMNIST' in settings.dataset:
        from utils.datasets.femnist import tf_femnist_train, tf_femnist_test, FEMNIST
        kwargs = {}
        flserver.set_dataset(FEMNIST, settings.data_path_prefix, kwargs, tf_femnist_train, tf_femnist_test)
        cnn_args.update({'num_classes': 62})
    elif 'TinyImageNet' in settings.dataset:
        from utils.datasets.tinyimagenet import TinyImageNetDataset, tf_tinyimagenet_test, tf_tinyimagenet_train
        kwargs = {}
        flserver.set_dataset(TinyImageNetDataset, settings.data_path_prefix, kwargs, tf_tinyimagenet_train, tf_tinyimagenet_test)
        cnn_args.update({'num_classes': 200})
    elif 'ImageNet' in settings.dataset:
        from utils.datasets.imagenet64 import ImageNet64, tf_imagenet64_test, tf_imagenet64_train
        kwargs = {}
        flserver.set_dataset(ImageNet64, settings.data_path_prefix, kwargs, tf_imagenet64_train, tf_imagenet64_test)
        cnn_args.update({'num_classes': 1000})
    else:
        raise NotImplementedError

    device_cnn_args.update(cnn_args)
    device_cnn_args_list = [deepcopy(device_cnn_args) for _ in range(settings.n_devices)]
    for idx, args in enumerate(device_cnn_args_list):
        if settings.algorithm != 'SLT':
            args.update({'subset_factor': flserver.memory_constraints_list[idx]})
    flserver.set_model([net for _ in range(settings.n_devices)], device_cnn_args_list)

    if settings.algorithm == 'FedAvg':
        # Evaluate on server with subset network
        args = deepcopy(cnn_args)
        args.update({'subset_factor': min(settings.memory_constraint)})
        flserver.set_model_evaluation(net_eval, args)
    elif settings.algorithm == 'HeteroFL' or settings.algorithm == 'FjORD':
        # Evaluation with largest device network
        args = deepcopy(cnn_args)
        args.update({'subset_factor': max(settings.memory_constraint)})
        flserver.set_model_evaluation(net_eval, args)
    elif settings.algorithm == 'SLT':
        args = deepcopy(cnn_args)
        flserver.set_model_evaluation(net_eval, args)
    else:
        # FD and FedRolex evaluate on server with Full NN (subset_factor=1.0)
        args = deepcopy(cnn_args)
        args.update({'subset_factor': 1.0})
        flserver.set_model_evaluation(net_eval, args)

    flserver.initialize()

    return flserver, run_path, run_hash, settings_as_dict


def fltraining(flserver, run_path, run_hash, settings_as_dict, settings):
    try:
        os.makedirs(run_path + run_hash)
        with open(run_path + run_hash + '/' + 'fl_setting.json', 'w') as fd:
            json.dump(settings_as_dict, fd, indent=4)
    except FileExistsError:
        pass

    logging.basicConfig(format='%(asctime)s - %(message)s',
                            filename=run_path + run_hash + '/run.log', level=logging.INFO, filemode='w')
    logging.info('Started')
    print(f'Settings Hash: {run_hash}')
    logging.info(f'Settings Hash: {run_hash}')
    logging.info('{' + '\n'.join('{!r}: {!r},'.format(k, v) for k, v in settings_as_dict.items()) + '}')

    with open(run_path + run_hash + '/' + 't_start.json', 'w') as fd:
        json.dump({'t_start': datetime.now().strftime('%Y_%M_%d_%H_%m_%s')}, fd, indent=4)

    if settings.plot is True:
        import visualization.plots as plots
        flserver.set_plotting_callback(plots.plot_config, run_path + run_hash)

    logging.info(f'memory_constraints_list: {flserver.memory_constraints_list}')
    try:
        flserver.run()
    except KeyboardInterrupt:
        pass

    if settings.plot is True:
        try:
            plots.plot_config(run_path + run_hash)
        except:
            print('Error plotting!')
            logging.info(f'Final plotting failed')
    try:
        os.unlink('latest_run')
    except FileNotFoundError:
        pass
    os.symlink(run_path + run_hash, 'latest_run')


def run(settings):
    flserver, run_path, run_hash, settings_as_dict = prepare(settings)
    if settings.dry_run is False:
        fltraining(flserver, run_path, run_hash, settings_as_dict, settings)
    else:
        print('DRY RUN PERFORMED SUCCESSFULLY')


if __name__ == '__main__':

    ALGORITHMS = ['SLT', 'FederatedDropout', 'FedAvg', 'FedRolex', 'HeteroFL', 'FjORD']
    DATASETS = ['CIFAR10', 'CIFAR100', 'FEMNIST', 'TinyImageNet', 'ImageNet']
    NNS = ['ResNet20', 'ResNet44', 'ResNet56', 'DenseNet40']
    DISTRIBUTIONS = ['iid', 'noniid']

    parser = argparse.ArgumentParser(description='FL Training')
    parser.add_argument('--session_tag', type=str, default='default_session', help='Sets name of subfolder for experiments')
    parser.add_argument('--algorithm', type=str, default=ALGORITHMS[0],
                        choices=ALGORITHMS, help=f'Choice of algorithm, available options are {ALGORITHMS}.')

    parser.add_argument('--dataset', type=str, choices=DATASETS, default=DATASETS[0], help=f'Choice of Datasets {DATASETS} (not all dataset/network combinations are possible). E.g., TinyImageNet and ImageNet64 are only compatible with ResNet44 and ResNet56.')
    parser.add_argument('--seed', type=int, default=11, help='Sets random seed for experiment')
    parser.add_argument('--network', choices=NNS, default=NNS[0], help=f'NN choice for experiement, options are {NNS}')
    parser.add_argument('--n_devices', type=int, default=100, help='Number of total FL devices.')
    parser.add_argument('--n_devices_per_round', type=int, default=10, help='Number of FL devices active in one round.')
    parser.add_argument('--data_distribution', type=str, choices=DISTRIBUTIONS, default=DISTRIBUTIONS[0], help=f'Sets type of data distribution used in the experiment. Choices are {DISTRIBUTIONS}. If noniid is chosen, the rate of noniid-nes can be controled with the dirichlet alpha parameter.')
    parser.add_argument('--distribution_dirichlet_alpha', type=float, default=0.1, help='Sets the dirichlet alpha parameter for noniid data distribution (controlls the rate of noniid-nes).')
    parser.add_argument('--lr', type=float, default=0.1, help='Starting learning rate (round 0).')
    parser.add_argument('--lr_reduction_factor', type=float, default=10.0, help='Final learning rate after decay (after the chosen number of FL rounds). The total number of FL rounds influce the learning rate at a given round.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for SGD optimizer.')
    parser.add_argument('--memory_constraint', nargs='+', type=float, default=[0.5], help=' Sets the memory constraint for the experiment. For ResNet structures 0.125, 0.25, and 0.5 is available, for DenseNet 0.33 and 0.66. To have heterogeneous constraints (groups) several constraint values can be chosen.')

    parser.add_argument('--n_rounds', type=int, default=1000, help='Number of total FL rounds')
    parser.add_argument('--torch_device', type=str, default='cuda:0', help='PyTorch device (cuda or cpu)')
    parser.add_argument('--progress_bar', type=bool, default=True, help='Progress bar printed in stdout.')
    parser.add_argument('--plot', type=bool, default=True, help='Plots are generated every 25 FL rounds as default behavior. Plotting can be disabled by setting it to False.')
    parser.add_argument('--data_path_prefix', type=str, default='data/', help='File location where the ML datasets are stored. Expects a trailing slash.')
    parser.add_argument('--dry_run', type=bool, default=False, help='Loads the NN, datasets, but does not apply training and does not create any files.')

    settings = parser.parse_args()
    run(settings)
