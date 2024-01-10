import torch
import copy


def submodel_indices(ratio, shape):
    indices = torch.arange(shape)
    start_idx = int(0)
    try:
        stop_idx = int(max(int(shape*((1)/((1/ratio)))), 1))
    except ZeroDivisionError:
        return indices[0:1]
    indices = indices[start_idx:stop_idx]
    return indices


def extract_submodel_resnet_structure(ratio, full_ratio, global_model, training_depth=1):
    '''
    ratio= is refered as the width of the NN (specifically the width
        that is used end-to-end). Goes from [0.0-1.0].
    full_ratio=is refered as the depth ratio of the NN (the part that
        is trained in full width). Goes from [0.0-1.0]
    '''
    submodel = copy.deepcopy(global_model)
    exceptions = ['num_batches_tracked']

    threshold = int(resnet_step_count(global_model)*full_ratio)

    count = 0
    used_ratio = 1.0
    freeze_list = []
    indices_in = None

    indices_dict = {}
    for key, item in submodel.items():
        if all(excluded_key not in key for excluded_key in exceptions):
            if count >= threshold:
                used_ratio = ratio
            else:
                freeze_list.append(key)
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
                    indices_out = submodel_indices(used_ratio, item.shape[0])
                    count += 1
                    submodel[key] = item[indices_out, :, :, :]
                    indices_dict.update({key: [indices_out]})
                    indices_in = indices_out
                elif 'shortcut.conv' in key:
                    indices_sc_out = indices_dict[key.replace('shortcut.conv', 'conv2')][0]
                    indices_sc_in = indices_dict[key.replace('shortcut.conv', 'conv1')][1]
                    item = item[indices_sc_out, :, :, :]
                    submodel[key] = item[:, indices_sc_in, :, :]
                    indices_in = indices_sc_out
                    indices_dict.update({key: [indices_sc_out, indices_sc_in]})
                else:
                    indices_out = submodel_indices(used_ratio, item.shape[0])
                    count += 1
                    out_shape = item.shape[0]
                    item = item[indices_out, :, :, :]

                    submodel[key] = item[:, indices_in, :, :]
                    indices_dict.update({key: [indices_out, indices_in]})
                    indices_in = indices_out

            # Case for Fully Connected Layers
            elif len(item.shape) == 2:
                flat_feature_map_size = int(item.shape[1]//out_shape)

                fc_indices = []
                for i in list(indices_in):
                    fc_indices += [int(i)*flat_feature_map_size + j for j in range(flat_feature_map_size)]

                submodel[key] = item[:, fc_indices]
                indices_dict.update({key: [torch.tensor(fc_indices)]})
            else:
                raise NotImplementedError

    dropped_layers = 0
    while True:
        if len(freeze_list) != 0:
            if 'conv' in freeze_list[-1]:
                freeze_list.pop()
                dropped_layers += 1
                if dropped_layers == training_depth:
                    break
            else:
                freeze_list.pop()
        else:
            break

    freeze_list = list(filter(lambda key: 'linear' not in key, freeze_list))
    return submodel, indices_dict, freeze_list


def resnet_step_count(global_model):
    submodel = copy.deepcopy(global_model)
    exceptions = ['num_batches_tracked']

    count = 0
    for key, item in submodel.items():
        if all(excluded_key not in key for excluded_key in exceptions):
            if len(item.shape) == 4:
                # Special Case for first Conv2d input Layer
                if item.shape[1] == 3:
                    count += 1
                elif 'shortcut.conv' in key:
                    pass
                else:
                    count += 1
    return count


def extract_submodel_densenet_structure(ratio, full_ratio, global_model, training_depth=1):
    '''
    ratio= is refered as the width of the NN (specifically the width
        that is used end-to-end). Goes from [0.0-1.0].
    full_ratio=is refered as the depth ratio of the NN (the part that
        is trained in full width). Goes from [0.0-1.0]
    '''
    submodel = copy.deepcopy(global_model)
    exceptions = ['num_batches_tracked']

    threshold = int(densenet_step_count(global_model)*full_ratio)
    count = 0
    used_ratio = 1.0
    freeze_list = []
    indices_in = None

    channels = []
    out_shape = submodel['bn.weight'].shape[0]
    skip_shape = submodel['dense1.0.conv2.weight'].shape[0]

    indices_dict = {}
    for key, item in submodel.items():
        if all(excluded_key not in key for excluded_key in exceptions):
            if count >= threshold:
                used_ratio = ratio
            else:
                freeze_list.append(key)
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
                    indices_out = submodel_indices(used_ratio, item.shape[0])
                    count += 1
                    submodel[key] = item[indices_out, :, :, :]
                    indices_dict.update({key : [indices_out]})
                    indices_in = indices_out
                    channels = indices_out
                else:
                    indices_out = submodel_indices(used_ratio, item.shape[0])
                    count += 1

                    if 'conv2' in key:
                        item = item[indices_out, :, :, :]
                        submodel[key] = item[:, indices_in, :, :]
                        indices_dict.update({key: [indices_out, indices_in]})
                        indices_in = torch.concat([indices_out, skip_shape + channels])

                    else:
                        item = item[indices_out, :, :, :]
                        submodel[key] = item[:, indices_in, :, :]
                        indices_dict.update({key: [indices_out, indices_in]})
                        indices_in = indices_out

                    if 'conv1' in key:
                        channels = indices_dict[key][1]

            # Case for Fully Connected Layers
            elif len(item.shape) == 2:
                flat_feature_map_size = int(item.shape[1]//out_shape)

                fc_indices = []
                for i in list(indices_in):
                    fc_indices += [int(i)*flat_feature_map_size + j for j in range(flat_feature_map_size)]

                submodel[key] = item[:, fc_indices]
                indices_dict.update({key: [torch.tensor(fc_indices)]})
            else:
                raise NotImplementedError

    dropped_layers = 0
    while True:
        if len(freeze_list) != 0:
            if 'conv' in freeze_list[-1]:
                freeze_list.pop()
                dropped_layers += 1
                if dropped_layers == training_depth:
                    break
            else:
                freeze_list.pop()
        else:
            break

    return submodel, indices_dict, freeze_list


def densenet_step_count(global_model):
    '''
    ratio= is refered as the width of the NN (specifically the width
        that is used end-to-end). Goes from [0.0-1.0].
    full_ratio=is refered as the depth ratio of the NN (the part that
        is trained in full width). Goes from [0.0-1.0]
    '''
    submodel = copy.deepcopy(global_model)
    exceptions = ['num_batches_tracked']

    count = 0
    for key, item in submodel.items():
        if all(excluded_key not in key for excluded_key in exceptions):
            if len(item.shape) == 4:
                # Special Case for first Conv2d input Layer
                if item.shape[1] == 3:
                    count += 1
                else:
                    count += 1
    return count
