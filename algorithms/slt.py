
from .fedavg import FedAvgEvaluation
from .fd import SubsetDevice, FederatedDropoutServer
import torch
import copy
import logging
import numpy as np
import json


class SLTServer(FederatedDropoutServer):
    _device_class = SubsetDevice
    _device_evaluation_class = FedAvgEvaluation
    last_indices = None

    def initialize(self):
        super().initialize()
        self._measurements_dict['level'] = []
        with open(self.nn_configs_path, 'r') as fd:
            try:
                self._nn_configs = json.load(fd)[str(min(self.memory_constraints_list))]
            except KeyError:
                raise NotImplementedError('Configuration not available for SLT')
        self._switching_idx = 0

        # Calculate switching points based on total rounds and params
        configs = self._nn_configs['values']
        trainable_parameters_list = []
        trainable_parameters = 0
        for config in configs:
            model, _, freeze_dict = self.extract_fnc(config[0], config[1], self._global_model)

            # Drop parameters that are already frozen
            for key in freeze_dict:
                model.pop(key)
            trainable_parameters = self.count_data_footprint(model)
            trainable_parameters_list.append(trainable_parameters)

        trainable_parameters = np.array(trainable_parameters_list)
        trainable_parameters = trainable_parameters/np.max(trainable_parameters)
        ranges = np.cumsum(trainable_parameters)
        ranges = ranges/np.max(ranges)

        # Pretraining rounds
        pretraining_rounds = int(configs[0][0]*self.n_rounds)

        logging.info(f'[SLT]: Pretraining rounds: {pretraining_rounds}')
        ranges = np.array(ranges*(self.n_rounds - pretraining_rounds), dtype=int).tolist()
        self._ranges = [pretraining_rounds] + [item + pretraining_rounds for item in ranges]

        # Saving values for later usage
        self._configs = [[configs[0][0], 0.0]] + configs

        if len(self._configs) > len(self._ranges):
            raise ValueError(f'ranges are of length {len(self._ranges)}, but configs are {len(self._configs)}')
        logging.info(f'[SLT]: Calculated Switching Points {self._ranges}')
        return

    def pre_round(self, round_n, rng):
        rand_selection_idxs = self.random_device_selection(self.n_devices, self.n_devices_per_round, rng)

        # Determine current configuration
        for ranges_idx, _ in enumerate(self._ranges):
            if round_n < self._ranges[ranges_idx]:
                break
            ranges_idx = min(ranges_idx, len(self._configs) - 1)

        # Extraction of trainable model out of the full model
        device_model_list = []
        list_of_indices_dict = []
        frozen_list = []

        for _, (dev_index) in enumerate(rand_selection_idxs):
            # Determin training depth
            memory_constraint = self.memory_constraints_list[dev_index]

            # Homogeneous case, all devices have the same memory constraint
            # Devices therefore train with training depth 1 (filling up *one* layer at a time)
            if memory_constraint == min(self.memory_constraints_list):
                training_depth = 1
            # If its the first round, in all cases devices train with depth one
            elif ranges_idx == 0:
                training_depth = 1
            # In case of heterogeneous constraints, the training depth is pulled from the nn_configuration LUT
            else:
                training_depth = self._nn_configs[str(memory_constraint)]['training_depth'][ranges_idx - 1]

            device_model, indices_dict, frozen = self.extract_fnc(self._configs[ranges_idx][0], self._configs[ranges_idx][1],
                                                                    self._global_model, training_depth=training_depth)
            device_model.update({'frozen': copy.deepcopy(frozen)})

            self._current_config = self._configs[ranges_idx]

            device_model_list.append(device_model)
            list_of_indices_dict.append(indices_dict)
            frozen_list.append(frozen)

        if ranges_idx != self._switching_idx:
            logging.info(f'[SLT]: Switching model at round {round_n} [{self._configs[ranges_idx][0]} {self._configs[ranges_idx][1]}]')

        self._switching_idx = ranges_idx
        self._list_of_indices_dict = list_of_indices_dict
        self._frozen_list = frozen_list
        return rand_selection_idxs, device_model_list

    def measure_data_upload(self, round_n, idxs):
        used_devices = [self._devices_list[i] for i in idxs]

        # Couting device upload
        byte_count = 0
        for idx, device in enumerate(used_devices):
            state_dict = device.get_model_state_dict()
            for key in state_dict:

                # Frozen layers do not have to be uploaded
                if any(key.startswith(k) for k in self._frozen_list[idx]):
                    continue
                param = state_dict[key]
                if isinstance(param, torch.Tensor):
                    # Assuming float32 (througout all algorithms)
                    val = 4
                    for i in range(len(param.shape)):
                        val *= param.shape[i]
                byte_count += val
        self._measurements_dict['data_upload'].append([byte_count, round_n])

    def measure_accuracy(self, round_n):
        # Evaluation of averaged global model
        self._evaluation_device.init_model()
        self._evaluation_device.set_model_state_dict(self.extract_fnc(self._current_config[0], self._current_config[1], copy.deepcopy(self._global_model))[0], strict=False)

        self._evaluation_device.test()
        accuracy = round(float(self._evaluation_device.accuracy_test), 4)
        logging.info(f'[SLT]: Round: {round_n} Test accuracy: {accuracy}')
        self._measurements_dict['accuracy'].append([accuracy, round_n])

        # Add Switching level
        self._measurements_dict['level'].append([self._switching_idx, round_n])

    def post_round(self, round_n, idxs):
        used_devices = [self._devices_list[i] for i in idxs]

        # Reference Model for averaging (from evaluation device)
        eval_model = self._global_model

        # Extract individual Device models with stored indices
        device_models = []
        device_masks = []

        for idx, device in enumerate(used_devices):
            model, mask = self.embedd_submodel(self._list_of_indices_dict[idx], device.get_model_state_dict(), eval_model)
            device_models.append(model)
            device_masks.append(mask)

        # Model Averaging based on extracted local models
        averaged_model = self.model_averaging(device_models, device_masks, eval_device_dict=eval_model)

        # Setting new global model
        self._global_model = averaged_model
