from .fedavg import FedAvgEvaluation
from .fd import SubsetDevice, FederatedDropoutServer

import logging
import copy


class HeteroFLServer(FederatedDropoutServer):
    _device_class = SubsetDevice
    _device_evaluation_class = FedAvgEvaluation

    def initialize(self):
        super().initialize()

        # Make sure that global model has full size
        # Only largest device model gets evaluated
        kwargs = copy.deepcopy(self._model_evaluation_kwargs)
        kwargs.update({'subset_factor': 1.0})
        self._global_model = self._model_evaluation(**kwargs).state_dict()

    def measure_accuracy(self, round_n):
        # Evaluation of averaged global model
        self._evaluation_device.init_model()

        # HeteroFL evaluates using the largest device model
        evaluation_state_dict, _ = self.extract_fnc(max(self.memory_constraints_list), self._global_model)
        self._evaluation_device.set_model_state_dict(evaluation_state_dict)

        self._evaluation_device.test()
        accuracy = round(float(self._evaluation_device.accuracy_test), 4)
        logging.info(f"[HeteroFL]: Round: {round_n} Test accuracy: {accuracy}")
        self._measurements_dict['accuracy'].append([accuracy, round_n])
