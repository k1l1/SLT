from abc import ABC
import torch
import sys
import numpy as np

import copy
import tqdm
import json
import random
import logging
import math


class FedAvgDevice(ABC):

    def __init__(self, device_id, storage_path):
        self._device_id = device_id
        self._storage_path = storage_path

        # Model related
        self._model = None
        self._model_kwargs = None
        self._model_class = None

        # Data related
        self._test_data = None
        self._train_data = None
        self._batch_size_test = 1024
        self._batch_size_train = 32

        # Training related
        self._optimizer = None
        self._optimizer_args = None
        self._loss_F = None
        self.lr = -1.0
        self.training_loss = None

        self._torch_device = None
        self._accuracy_test = None

    def set_seed(self, seed):
        torch.manual_seed(seed)

    def set_model(self, model_list, kwargs_list):
        self._model_kwargs = kwargs_list
        self._model_class = model_list

    def init_model(self):
        self._model = self._model_class(**self._model_kwargs)

    def del_model(self):
        self._model = None

    def set_test_data(self, dataset):
        self._test_data = dataset
        return

    def set_train_data(self, dataset):
        self._train_data = dataset
        return

    def set_torch_device(self, torch_dev):
        self._torch_device = torch_dev

    def set_optimizer(self, optimizer, optimizer_args):
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args

    def set_loss_function(self, loss_F):
        self._loss_F = loss_F

    def get_model_state_dict(self):
        assert self._model is not None, 'Device has no NN model'
        return self._model.state_dict()

    def set_model_state_dict(self, model_dict, strict=True):
        self._model.load_state_dict(copy.deepcopy(model_dict), strict=strict)

    def return_reference(self):
        return self

    def _check_trainable(self):
        assert self._model is not None, 'device has no NN model'
        assert self._train_data is not None, 'device has no training dataset'
        assert self._torch_device is not None, 'No torch_device is set'
        assert self._optimizer is not None, 'No optimizer is set'
        assert self._loss_F is not None, 'No loss function is set'

    def _check_testable(self):
        assert self._model is not None, 'device has no NN model'
        assert self._test_data is not None, 'device has no test dataset'
        assert self._torch_device is not None, 'No torch_device is set'

    @staticmethod
    def correct_predictions(labels, outputs):
        res = (torch.argmax(outputs.cpu().detach(), axis=1) ==
              labels.cpu().detach()).sum()
        return res

    @staticmethod
    def assert_if_nan(*tensors):
        for tensor in tensors:
            assert not torch.isnan(tensor).any(), 'Loss is NaN'
 
    def _train(self, n_epochs=1):
        self._model.to(self._torch_device)
        self._model.train()

        loss_function = self._loss_F()

        trainloader = torch.utils.data.DataLoader(self._train_data,
                                    batch_size=self._batch_size_train,
                                    shuffle=True, pin_memory=True)

        recorded_loss = []
        optimizer = self._optimizer([param for param in self._model.parameters() if param.requires_grad], lr=self.lr, **self._optimizer_args)
        for _ in range(n_epochs):
            for _, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self._torch_device), labels.to(self._torch_device)
                optimizer.zero_grad()
                output = self._model(inputs)
                loss = loss_function(output, labels)
                self.assert_if_nan(loss)
                loss.backward()
                recorded_loss.append(loss.cpu().detach().item())
                optimizer.step()
        self.training_loss = torch.mean(torch.tensor(recorded_loss))

    def device_round(self):
        self._check_trainable()
        self._train()
        self._model.to('cpu')
        return self


class FedAvgEvaluation(FedAvgDevice):
    def _test(self):
        self._model.to(self._torch_device)
        self._model.eval()

        assert not self._test_data.dataset.train, 'Wrong dataset for testing.'

        testloader = torch.utils.data.DataLoader(self._test_data, shuffle=True,
                                                batch_size=self._batch_size_test, pin_memory=True)
        correct_predictions = 0

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(self._torch_device), labels.to(self._torch_device)
                output = self._model(inputs)
                correct_predictions += self.correct_predictions(labels, output)
            self.accuracy_test = correct_predictions/len(self._test_data)
        return

    def test(self):
        self._check_testable()
        self._test()
        self._model.to('cpu')
        return self

    def device_round(self):
        raise NotImplementedError('Evaluation Device does not have a round function.')


class FedAvgServer(ABC):

    _device_evaluation_class = FedAvgEvaluation
    _device_class = FedAvgDevice

    def __init__(self, storage_path):
        self._devices_list = []
        self._storage_path = storage_path

        # General
        self.torch_device = None
        self.n_rounds = 0

        # Devices
        self.n_devices_per_round = 0
        self.n_devices = 0
        self._device_constraints = None
        self._global_model = None

        # Debug related
        self._plotting_f = None
        self._plotting_arg = None
        self.progress_output = True
        self.n_rounds_between_plot = 25

        self._measurements_dict = {}
        self._measurements_dict['accuracy'] = []
        self._measurements_dict['data_upload'] = []
        self._measurements_dict['learning_rate'] = []
        self._measurements_dict['training_loss'] = []

        # Training related
        self._optimizer = None
        self._optimizer_args = None
        self.lr = - 1.0
        self.lr_min = - 1.0

        # Data related
        self._test_data = None
        self._train_data = None
        self.split_f = None

        self._seed_n = 0

    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        self._seed_n = seed

    def set_optimizer(self, optimizer, optimizer_args):
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args

    def set_plotting_callback(self, f, arg):
        self._plotting_f = f
        self._plotting_arg = arg

    @staticmethod
    def random_device_selection(n_devices, n_devices_per_round, rng):
        dev_idxs = rng.permutation(n_devices)[0:n_devices_per_round].tolist()
        return dev_idxs

    @staticmethod
    def count_data_footprint(state_dict):
        counted_bytes = 0
        for key in state_dict:
            param = state_dict[key]
            if isinstance(param, torch.Tensor):
                val = 4
                for i in range(len(param.shape)):
                    val *= param.shape[i]
                counted_bytes += val
        return counted_bytes

    def initialize(self):

        idxs_list = self.split_f(self._train_data, self.n_devices)

        self._evaluation_device = self._device_evaluation_class(0, self._storage_path)
        self._evaluation_device.set_seed(self._seed_n)
        self._evaluation_device.set_model(self._model_evaluation, self._model_evaluation_kwargs)
        self._evaluation_device.init_model()
        self._evaluation_device.set_test_data(self._test_data)
        self._evaluation_device.set_torch_device(self.torch_device)
        self._evaluation_device._batch_size_test = 1024

        self._devices_list = [self._device_class(i, self._storage_path) for i in range(self.n_devices)]

        for i, device in enumerate(self._devices_list):
            device.set_seed(self._seed_n)
            device.set_model(self._model[i], self._model_kwargs[i])
            device.set_train_data(torch.utils.data.Subset(self._train_data.dataset, idxs_list[i]))
            device.set_loss_function(torch.nn.CrossEntropyLoss)
            device.set_optimizer(self._optimizer, self._optimizer_args)
            device.set_torch_device(self.torch_device)

        # Get initial RAW NN state dict from evaluation device
        self._global_model = copy.deepcopy(self._evaluation_device._model.state_dict())
        return

    def save_dict_to_json(self, filename, input_dict):
        with open(self._storage_path + '/' + filename, 'w') as fd:
            json.dump(input_dict, fd, indent=4)

    def set_dataset(self, dataset, path,  kwargs, transform_train, transform_test):
        kwargs.update({'transform' : transform_train})
        data_train = dataset(path, train=True, **kwargs)

        kwargs = copy.deepcopy(kwargs)
        kwargs.update({'transform' : transform_test})
        data_test = dataset(path, train=False, **kwargs)

        self._train_data = torch.utils.data.Subset(data_train, torch.arange(0, len(data_train)))
        self._test_data = torch.utils.data.Subset(data_test, torch.arange(0, len(data_test)))
        return

    def set_model(self, model_list, kwargs_list):
        self._model = model_list
        self._model_kwargs = kwargs_list

    def set_model_evaluation(self, model, kwargs):
        self._model_evaluation = model
        self._model_evaluation_kwargs = kwargs

    def learning_rate_scheduling(self, round_n, n_rounds):
        lr = self.lr_min + 0.5*(self.lr - self.lr_min)*(1 + math.cos(math.pi*round_n/n_rounds))
        return lr

    def measure_data_upload(self, round_n, idxs):
        used_devices = [self._devices_list[i] for i in idxs]

        # Couting device upload
        byte_count = 0
        for device in used_devices:
            state_dict = device.get_model_state_dict()
            for key in state_dict:
                param = state_dict[key]
                if isinstance(param, torch.Tensor):
                    val = 4
                    for i in range(len(param.shape)):
                        val *= param.shape[i]
                byte_count += val
        self._measurements_dict['data_upload'].append([byte_count, round_n])

    def measure_accuracy(self, round_n):
        # Evaluation of averaged global model
        self._evaluation_device.set_model_state_dict(copy.deepcopy(self._global_model), strict=False)
        self._evaluation_device.test()
        accuracy = round(float(self._evaluation_device.accuracy_test), 4)
        logging.info(f'[FEDAVG]: Round: {round_n} Test accuracy: {accuracy}')
        self._measurements_dict['accuracy'].append([accuracy, round_n])

    def measure_loss(self, round_n, idxs):
        loss = []
        for dev_idx in idxs:
            loss.append(self._devices_list[dev_idx].training_loss)
        loss = float(torch.mean(torch.tensor(loss)))
        self._measurements_dict['training_loss'].append([loss, round_n])

    def pre_round(self, round_n, rng):
        rand_selection_idxs = self.random_device_selection(self.n_devices, self.n_devices_per_round, rng)
        return rand_selection_idxs, [self._global_model for _ in range(self.n_devices_per_round)]

    def post_round(self, round_n, idxs):
        used_devices = [self._devices_list[i] for i in idxs]
        averaged_model = self.model_averaging([dev.get_model_state_dict() for dev in used_devices],
                                                eval_device_dict=self._global_model)
        self._global_model = averaged_model

    def run(self):
        self.check_device_data()
        print(f'#Samples on devices: {[len(dev._train_data) for dev in self._devices_list]}')
        logging.info(f'[FL_BASE]: #Samples on devices: {[len(dev._train_data) for dev in self._devices_list]}')

        # Plot Data Distribution
        if self.split_f.is_plot:
            self.split_f.plot_distribution()

        rng = np.random.default_rng(self._seed_n)

        tbar = tqdm.tqdm(iterable=range(self.n_rounds), total=self.n_rounds, file=sys.stdout, disable=not self.progress_output)
        for round_n in tbar:

            # Learning rate scheduling
            lr = self.learning_rate_scheduling(round_n, self.n_rounds)

            # Selection of devices
            idxs, device_models = self.pre_round(round_n, rng)

            # Init NN models
            for i, dev_idx in enumerate(idxs):
                self._devices_list[dev_idx].lr = lr
                self._devices_list[dev_idx].init_model()
                self._devices_list[dev_idx].set_model_state_dict(device_models[i])

            # Local device training
            for dev in [self._devices_list[i] for i in idxs]:
                dev.device_round()

            # Measure data uplaod comming from devices
            self.measure_data_upload(round_n, idxs)

            # Knwoledge aggregation // global model gets set
            self.post_round(round_n, idxs)

            # Measure accuracy
            self.measure_accuracy(round_n)

            # Delete models (stateless devices)
            for dev_idx in idxs:
                self._devices_list[dev_idx].del_model()

            # Get loss from devices and add it to measurements
            self.measure_loss(round_n, idxs)

            # Add learning rate to measurements
            self._measurements_dict['learning_rate'].append([lr, round_n])

            # Save accuracy dict
            self.save_dict_to_json('measurements.json', self._measurements_dict)

            if self.progress_output:
                tbar.set_description(f"round_n {round_n}, acc: {self._measurements_dict['accuracy'][round_n][0]}")
            else:
                print(f"round_n {round_n}, acc={self._measurements_dict['accuracy'][round_n][0]}")

            # Plotting
            if (round_n % self.n_rounds_between_plot) == 0 and round_n != 0:
                if self._plotting_f is not None:
                    try:
                        self._plotting_f(self._plotting_arg)
                    except Exception as e:
                        print(f'Error plotting! {e}')

    def check_device_data(self):
        for i in range(len(self._devices_list)):
            for j in range(len(self._devices_list)):
                if i != j:
                    assert set(self._devices_list[i]._train_data.indices.tolist()).isdisjoint(set(
                            self._devices_list[j]._train_data.indices.tolist())), 'Devices do not exclusivly have access to their data!'

    @staticmethod
    def model_averaging(list_of_state_dicts, eval_device_dict=None):
        averaging_exceptions = ['num_batches_tracked']

        averaged_dict = copy.deepcopy(list_of_state_dicts[0])
        for key in averaged_dict:
            if all(module_name not in key for module_name in averaging_exceptions):
                averaged_dict[key] = torch.mean(torch.stack([state_dict[key]
                                        for state_dict in list_of_state_dicts]), dim=0)

        averaged_dict = {k: v for k, v in averaged_dict.items() if all(module_name not in k for module_name in averaging_exceptions)}
        return averaged_dict
