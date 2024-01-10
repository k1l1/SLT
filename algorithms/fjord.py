from .fedavg import FedAvgEvaluation

from .fd import SubsetDevice
from .heterofl import HeteroFLServer

import torch
import copy
import logging
import numpy as np


class FjordDevice(SubsetDevice):
    fjord_p = None
    fjord_rng = None

    def _train(self, n_epochs=1):
        kwargs = copy.deepcopy(self._model_kwargs)

        maximum_subset_factor = kwargs['subset_factor']
        trainable_subset_factors = [subset_factor for subset_factor in self.fjord_p if subset_factor <= maximum_subset_factor]

        logging.info(f'[FJORD]: device {self._device_id} k_max: {maximum_subset_factor} trainable: {trainable_subset_factors}')

        loss_function = self._loss_F()
        model = None

        trainloader = torch.utils.data.DataLoader(self._train_data,
                                    batch_size=self._batch_size_train,
                                    shuffle=True, pin_memory=True)
        recorded_loss = []
        for _ in range(n_epochs):
            for batch_idx, (inputs, labels) in enumerate(trainloader):

                if batch_idx != 0:
                    # Fuse model
                    current_state_dict = model.state_dict()
                    self._model.load_state_dict(self.fuse_state_dicts(self._model.state_dict(),
                                                                      current_state_dict))

                # Randomly select a subset_factor for next mini-batch
                rnd_idx = self.fjord_rng.integers(0, len(trainable_subset_factors))
                subset_factor = trainable_subset_factors[rnd_idx]
                kwargs.update({'subset_factor': subset_factor})

                # Set model
                model = self._model_class(**kwargs)

                model.load_state_dict(self.extract_submodel(self._model.state_dict(), model.state_dict()))
                model.to(self._torch_device)
                model.train()

                # Initialize optimizer
                optimizer = self._optimizer(model.parameters(), lr=self.lr, **self._optimizer_args)

                inputs, labels = inputs.to(self._torch_device), labels.to(self._torch_device)

                optimizer.zero_grad()
                output = model(inputs)

                loss = loss_function(output, labels)
                self.assert_if_nan(loss)
                loss.backward()
                recorded_loss.append(loss.cpu().detach().item())
                optimizer.step()

        current_state_dict = model.state_dict()
        self._model.load_state_dict(self.fuse_state_dicts(
                            self._model.state_dict(), current_state_dict))

        self.training_loss = torch.mean(torch.tensor(recorded_loss))

    @staticmethod
    def fuse_state_dicts(state_dict_target, state_dict_source):
        for key in state_dict_target:
            item = state_dict_source[key]

            if len(item.size()) == 4:
                state_dict_target[key][0:item.shape[0], 0:item.shape[1], :, :] = item
            elif len(item.size()) == 2:
                state_dict_target[key][0:item.shape[0], 0:item.shape[1]] = item
            elif len(item.size()) == 1:
                state_dict_target[key][0:item.shape[0]] = item
            else:
                raise NotImplementedError
        return state_dict_target

    @staticmethod
    def extract_submodel(state_dict, reference_state_dict):
        state_dict = copy.deepcopy(state_dict)

        for key in state_dict:
            target_shape = reference_state_dict[key].shape

            if len(target_shape) == 4:
                state_dict[key] = state_dict[key][0:target_shape[0], 0:target_shape[1], :, :]
            elif len(target_shape) == 2:
                state_dict[key] = state_dict[key][0:target_shape[0], 0:target_shape[1]]
            elif len(target_shape) == 1:
                state_dict[key] = state_dict[key][0:target_shape[0]]
            else:
                raise NotImplementedError
        return state_dict


class FjordServer(HeteroFLServer):
    _device_class = FjordDevice
    _device_evaluation_class = FedAvgEvaluation
    _fjord_rng = None

    def initialize(self):
        super().initialize()

        # Random generator for FjORD
        self._fjord_rng = np.random.default_rng(self._seed_n)

        # Set Fjord levels
        subset_factor_levels = list(set([kwargs['subset_factor'] for kwargs in self._model_kwargs]))

        for device in self._devices_list:
            device.fjord_p = copy.deepcopy(subset_factor_levels)

            # Each device uses the same rng (otherwise all devices
            # switch exactly the same)
            device.fjord_rng = self._fjord_rng
