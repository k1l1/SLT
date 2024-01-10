import copy
import torch


def submodel_indices(ratio, shape):
    indices = torch.randperm(shape)
    indices = indices[0:int(ratio*shape)]
    return indices


def extract_submodel_resnet_structure(ratio, global_model, round_n=None):
    # Expects all entries of modules to be in execution order!
    submodel = copy.deepcopy(global_model)
    exceptions = ['num_batches_tracked']

    indices_in = None
    out_shape = None

    indices_dict = {}
    for key, item in submodel.items():
        if all(excluded_key not in key for excluded_key in exceptions):
            # Case for bias, running_mean and running_var
            if len(item.shape) == 1:
                # Output bias remains fully-sized
                if key == 'linear.bias':
                    continue
                else:
                    submodel[key] = item[indices_in]
                    indices_dict.update({key: [indices_in]})
            # Case for Conv2d
            elif len(item.shape) == 4:
                # Special Case for first Conv2d input Layer
                if item.shape[1] == 3:
                    indices_out = submodel_indices(ratio, item.shape[0])
                    submodel[key] = item[indices_out, :, :, :]
                    indices_dict.update({key: [indices_out]})
                    indices_in = indices_out
                # Special case for shortcut Conv layer, as the input has to follow the last
                # Blocks input and the output should be the same as the blocks conv2 output
                elif 'shortcut.conv' in key:
                    indices_sc_out = indices_dict[key.replace('shortcut.conv', 'conv2')][0]
                    indices_sc_in = indices_dict[key.replace('shortcut.conv', 'conv1')][1]
                    item = item[indices_sc_out, :, :, :]
                    submodel[key] = item[:, indices_sc_in, :, :]
                    indices_in = indices_sc_out
                    indices_dict.update({key: [indices_sc_out, indices_sc_in]})
                else:
                    # Special case for shortcut case, as the input has to follow the last
                    # Blocks input and the output should be the same as the blocks conv2 output
                    if 'conv2' in key and indices_dict[key.replace('conv2', 'conv1')][1].shape[0] == int(ratio*item.shape[0]):
                            indices_out = indices_dict[key.replace('conv2', 'conv1')][1]
                    else:
                        indices_out = submodel_indices(ratio, item.shape[0])
                    out_shape = item.shape[0]
                    item = item[indices_out, :, :, :]
                    submodel[key] = item[:, indices_in, :, :]
                    indices_dict.update({key: [indices_out, indices_in]})
                    indices_in = indices_out
            
            # Case for Fully Connected Layers
            elif len(item.shape) == 2:
                flat_feature_map_size = item.shape[1]//out_shape

                fc_indices = []
                for i in list(indices_in):
                    fc_indices += [i*flat_feature_map_size + j for j in range(flat_feature_map_size)]

                submodel[key] = item[:, fc_indices]
                indices_dict.update({key: [fc_indices]})
            else: 
                raise NotImplementedError
    return submodel, indices_dict


def extract_submodel_densenet_structure(ratio, global_model, round_n=None):
    # Expects all entries of modules to be in execution order!
    submodel = copy.deepcopy(global_model)
    exceptions = ['num_batches_tracked']

    indices_in = None
    out_shape = submodel['bn.weight'].shape[0]
    skip_shape = submodel['dense1.0.conv2.weight'].shape[0]

    indices_dict = {}
    for key, item in submodel.items():
        if all(excluded_key not in key for excluded_key in exceptions):
            # Case for bias, running_mean and running_var
            if len(item.shape) == 1:
                # Output bias remains fully-sized
                if key == 'linear.bias':
                    continue
                else:
                    submodel[key] = item[indices_in]
                    indices_dict.update({key: [indices_in]})
            # Case for Conv2d
            elif len(item.shape) == 4:

                # Special Case for first Conv2d input Layer
                if item.shape[1] == 3:
                    indices_out = submodel_indices(ratio, item.shape[0])
                    submodel[key] = item[indices_out, :, :, :]
                    indices_dict.update({key: [indices_out]})
                    indices_in = indices_out
                    channels = indices_out
                else:
                    # This ratio has to be fixed
                    if 'conv2' in key:
                        # Calculate skip ratio such that the number of remaining skip connection filters match exactly
                        skip_ratio = int((global_model[key.replace('conv2', 'conv1')].shape[1] + skip_shape)*ratio - int(global_model[key.replace('conv2', 'conv1')].shape[1]*ratio))
                        skip_ratio = skip_ratio / item.shape[0]
                        
                        indices_out = submodel_indices(skip_ratio, item.shape[0])
                        item = item[indices_out, :, :, :]
                        submodel[key] = item[:, indices_in, :, :]
                        indices_dict.update({key: [indices_out, indices_in]})
                        indices_in = torch.concat([indices_out, skip_shape + channels])
                    else:
                        indices_out = submodel_indices(ratio, item.shape[0])
                        item = item[indices_out, :, :, :]
                        submodel[key] = item[:, indices_in, :, :]
                        indices_dict.update({key: [indices_out, indices_in]})
                        indices_in = indices_out
                    if 'conv1' in key:
                        channels = indices_dict[key][1]

            # Case for Fully Connected Layers
            elif len(item.shape) == 2:
                flat_feature_map_size = item.shape[1]//out_shape

                fc_indices = []
                for i in list(indices_in):
                    fc_indices += [i*flat_feature_map_size + j for j in range(flat_feature_map_size)]

                submodel[key] = item[:, fc_indices]
                indices_dict.update({key: [fc_indices]})
            else:
                raise NotImplementedError
    return submodel, indices_dict
