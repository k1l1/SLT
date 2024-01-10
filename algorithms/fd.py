from algorithms.fedavg import FedAvgServer, FedAvgEvaluation, FedAvgDevice
import torch
import copy


class SubsetDevice(FedAvgDevice):
    subset_factor = -1.0


class FederatedDropoutServer(FedAvgServer):
    _device_class = SubsetDevice
    _device_evaluation_class = FedAvgEvaluation
    _list_of_indices_dict = None

    @staticmethod
    def embedd_submodel(indices_dict, submodel, global_model):
        model = copy.deepcopy(global_model)
        for key, item in model.items():
            item.zero_()
            pass

        mask = copy.deepcopy(model)
        for key, item in mask.items():
            if len(item.shape) == 4:
                mask[key] = item[:, :, 0, 0]

        exceptions = ['num_batches_tracked']
        for key, item in model.items():
            if all(excluded_key not in key for excluded_key in exceptions):
                # Case for bias, running_mean and running_var
                if len(item.shape) == 1:
                    # Output bias remains fully-sized
                    if key == 'linear.bias':
                        model[key] = submodel[key]
                        mask[key] = torch.ones(model[key].shape)
                    else:
                        indices = indices_dict[key][0]
                        model[key][indices] = submodel[key]
                        mask[key][indices] = 1
                # Case for Conv2d
                elif len(item.shape) == 4:

                    # Special Case for first Conv2d input Layer
                    if item.shape[1] == 3:
                        indices_out = indices_dict[key][0]
                        model[key][indices_out, :, :, :] = submodel[key]
                        mask[key][indices_out, :] = 1
                    else:
                        indices_in = indices_dict[key][1]
                        indices_out = indices_dict[key][0]
                        indices_out, indices_in = torch.meshgrid(indices_out, indices_in, indexing='ij')
                        model[key][indices_out, indices_in, :, :] = submodel[key]
                        mask[key][indices_out, indices_in] = 1

                # Case for Fully Connected Layers
                elif len(item.shape) == 2:
                    indices_in = indices_dict[key][0]
                    model[key][:, indices_in] = submodel[key]
                    mask[key][:, indices_in] = 1
                else:
                    raise NotImplementedError
        return model, mask

    @staticmethod
    def model_averaging(list_of_state_dicts, list_of_masks, eval_device_dict=None):
        averaging_exceptions = ['num_batches_tracked']

        averaged_dict = copy.deepcopy(eval_device_dict)
        for key in averaged_dict:
            if all(module_name not in key for module_name in averaging_exceptions):
                stacked_mask = torch.stack([mask[key] for mask in list_of_masks], dim=0)
                parameter_stack = torch.stack([sd[key] for sd in list_of_state_dicts], dim=0)
                mask = torch.sum(stacked_mask, dim=0)
                if len(averaged_dict[key].shape) == 4:
                    mask = mask[:, :, None, None].repeat(1, 1, averaged_dict[key].shape[2], averaged_dict[key].shape[3])
                averaged_parameter = torch.sum(parameter_stack, dim=0)/mask
                averaged_dict[key][mask != 0] = averaged_parameter[mask != 0]
                pass

        averaged_dict = {k: v for k, v in averaged_dict.items() if all(module_name not in k for module_name in averaging_exceptions)}

        return averaged_dict

    def pre_round(self, round_n, rng):
        rand_selection_idxs = self.random_device_selection(self.n_devices, self.n_devices_per_round, rng)

        # Extraction of trainable model out of the full model
        device_model_list = []
        list_of_indices_dict = []
        for _, (dev_index) in enumerate(rand_selection_idxs):

            device_model, indices_dict = self.extract_fnc(self.memory_constraints_list[dev_index], self._global_model, round_n=round_n)
            device_model_list.append(device_model)
            list_of_indices_dict.append(indices_dict)

        self._list_of_indices_dict = list_of_indices_dict
        return rand_selection_idxs, device_model_list

    def post_round(self, round_n, idxs):
        used_devices = [self._devices_list[i] for i in idxs]

        # Extract individual Device models with stored indices
        device_models = []
        device_masks = []

        for i, device in enumerate(used_devices):
            model, mask = self.embedd_submodel(self._list_of_indices_dict[i], device.get_model_state_dict(), self._global_model)
            device_models.append(model)
            device_masks.append(mask)

        # Model Averaging based on extracted local models
        averaged_model = self.model_averaging(device_models, device_masks,
                                              eval_device_dict=self._global_model)

        # Setting new global model
        self._global_model = averaged_model
