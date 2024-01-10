import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import os
import copy
import hashlib


def dict_hash(dictionary):
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


class RunConfig():
    _config = None
    _measurements = None
    cfg_hash = None

    def __init__(self, path_to_folder, property_string='accuracy'):
        with open(path_to_folder + '/fl_setting.json') as fd:
            self._config = (json.load(fd))

        self._property_string = property_string
        self.cfg_hash = dict_hash(self._config)
        self.path_to_folder = path_to_folder

        try:
            with open(path_to_folder + '/measurements.json') as fd:
                self._measurements = (json.load(fd))
        except FileNotFoundError:
            self._config = None

    def __hash__(self):
        return int(self.hash_except_for(['seed']), 16)

    def __getitem__(self, key):
        return self._config[key]

    def __iter__(self, key):
        return iter(self._config)

    def __contains__(self, x):
        return x in self._config

    def get_Y(self):
        if self._property_string == 'accuracy':
            return np.asarray(self._measurements[self._property_string])[:, 0]
        elif self._property_string == 'data_upload':
            return np.cumsum(np.asanyarray(self._measurements[self._property_string])[:, 0]/(10**9))
        elif self._property_string == 'learning_rate':
            return np.asarray(self._measurements[self._property_string])[:, 0]
        elif self._property_string == 'training_loss':
            return np.asarray(self._measurements[self._property_string])[:, 0]
        elif self._property_string == 'max_accuracy_over_data':
            return np.maximum.accumulate(np.asarray(self._measurements['accuracy'])[:, 0])
        else:
            raise NotImplementedError

    def get_X(self):
        if self._property_string == 'max_accuracy_over_data':
            return np.cumsum(np.asanyarray(self._measurements['data_upload'])[:, 0]/(10**9))
        else:
            return np.asarray(self._measurements[self._property_string])[:, 1]

    def get_max(self):
        return np.max(np.asarray(self._measurements['accuracy'])[:, 0])

    def keys(self):
        return self._config.keys()

    def __eq__(self, other):
        if self.hash_except_for(['seed']) == other.hash_except_for(['seed']):
            return True
        else:
            return False

    def hash_except_for(self, key_list):
        cfg = copy.deepcopy(self._config)
        for item in key_list:
            cfg.pop(item)
        return dict_hash(cfg)


def plot_property(main_run, list_of_other_runs, list_of_equal_keys, save_path='', property_string='accuracy'):

    runs = [main_run] + list_of_other_runs
    if list_of_equal_keys[0] not in main_run.keys():
        return
    runs = list(filter(lambda x: list_of_equal_keys[0] in x.keys(), runs))
    for key in list_of_equal_keys:
        runs = list(filter(lambda run: run[key] == main_run[key], runs))
    runs = list(sorted(runs, key=lambda run: run.get_max(), reverse=True))

    plt.gcf().clear()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    unique = list(set(runs))
    unique = list(sorted(unique, key=lambda run: run.get_max(), reverse=True))
    for unique_run in unique:
        descr = f'avg_max: {unique_run.get_max()}\n'
        for key in unique_run._config:
            for run in unique:
                if key not in run or (unique_run[key] != run[key] and key != property_string):
                    descr += f'{key} : {unique_run[key]} ' + '\n'
                    break
        Y = [run.get_Y() for run in runs if run == unique_run]
        measurements = [run.get_Y() for run in runs if run == unique_run]
        X = [run.get_X() for run in runs if run == unique_run]

        max_common_length = min([len(item) for item in Y])
        measurements = [item[0:max_common_length] for item in measurements]

        # Test validity of fill_between plot
        X_common_lenght = np.array([item[0:max_common_length] for item in X])

        is_accurate = True
        for i in range(X_common_lenght.shape[0]):
            for j in range(X_common_lenght.shape[0]):
                if i != j:
                    if not np.all(np.isclose(X_common_lenght[i], X_common_lenght[j])):
                        is_accurate = False
                        break
            if not is_accurate:
                break

        if not is_accurate:
            plt.text(0.01, 0.925, 'WARNING: fill_between not accurate',
                     ha='left', va='bottom', transform=plt.gca().transAxes,
                     color='red', fontsize=12)

        n_seeds = len(measurements)

        avg = np.mean(measurements, axis=0)
        std = np.std(measurements, axis=0)

        if len(measurements) > 1:
            if unique_run == main_run:
                p = ax.plot(main_run.get_X(), main_run.get_Y(), linewidth=3.0, label=descr)
            else:
                p = ax.plot(0, 0, label=descr)
            ax.fill_between(X[0][0:max_common_length],
                            avg + std, avg-std, alpha=0.25, color=p[0].get_color())
        else:
            linewidth = (3.0 if unique_run == main_run else 1.0)
            ax.plot(unique_run.get_X(), unique_run.get_Y(), linewidth=linewidth, label=descr)
        pass

        if unique_run == main_run:
            with open(save_path + f'unique_{property_string}.csv', 'w') as fd:
                writer = csv.writer(fd, delimiter=',')
                writer.writerow(['round', 'avgp', 'avgm', 'avg', 'n_seeds'])
                rounds = main_run.get_X()
                if len(rounds) == len(avg):
                    for idx in range(len(avg)):
                        writer.writerow([rounds[idx],
                                         round(100*(avg[idx] + std[idx]), 3),
                                         round(100*(avg[idx] - std[idx]), 3),
                                         round(100*(avg[idx]), 3),
                                         n_seeds])

            with open(save_path + f'unique_{property_string}.step50.csv', 'w') as fd:
                writer = csv.writer(fd, delimiter=',')
                writer.writerow(['round', 'avgp', 'avgm', 'avg', 'n_seeds'])
                rounds = main_run.get_X()
                if len(rounds) == len(avg):
                    for idx in range(0, len(avg), 50):
                        writer.writerow([rounds[idx],
                                round(100*(avg[idx] + std[idx]), 3),
                                round(100*(avg[idx] - std[idx]), 3),
                                round(100*(avg[idx]), 3), n_seeds])
                    if len(avg) > 1:
                        writer.writerow([rounds[len(avg) - 1],
                                round(100*(avg[len(avg) - 1] + std[len(avg) - 1]), 3),
                                round(100*(avg[len(avg) - 1] - std[len(avg) - 1]), 3),
                                round(100*(avg[len(avg) - 1]), 3), n_seeds])
                   
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15))

    title_str = ''
    for key in list_of_equal_keys:
        title_str += f'{key} : {main_run[key]} \n'

    ax.set_title(title_str)
    text = ax.text(-0.3, 1.05, ' ', transform=ax.transAxes)
    ax.grid('on')
    if property_string == 'max_accuracy_over_data':
        ax.set_ylabel('Cummulative Maximum Accuracy')
        ax.set_xlabel('Required Communication in Gigabyte')
    elif property_string == 'data_upload':
        ax.set_ylabel('Required Communication in Gigabyte')
        ax.set_xlabel('FL rounds')
    else:
        ax.set_ylabel(property_string)
        ax.set_xlabel('FL rounds')

    if property_string == 'data_upload':
        ax.set_yscale('log')
        ax.grid(which='both', axis='y')
    if property_string == 'max_accuracy_over_data':
        ax.set_xscale('log')
        ax.grid(which='minor', axis='x')
    save_str = property_string + '_'
    for key in list_of_equal_keys:
        save_str += str(key) + '_'

    fig.savefig(save_path + f'{save_str}.png', bbox_extra_artists=(lgd, text), bbox_inches='tight')
    pass


def plot_config(run_path):

    path_prefix = run_path.split('run_')[0]

    main_run = RunConfig(run_path, property_string='accuracy')
    run_paths = os.listdir(path_prefix)
    run_paths = list(filter(lambda x: x.startswith('run_'), run_paths))
    run_cfg_list = [RunConfig(path_prefix + path) for path in run_paths if RunConfig(path_prefix + path).cfg_hash != main_run.cfg_hash]
    run_cfg_list = list(filter(lambda x: x._config is not None, run_cfg_list))

    plot_property(main_run, run_cfg_list, ['session_tag'], save_path=run_path + '/', property_string='accuracy')
    plot_property(main_run, run_cfg_list, ['algorithm'], save_path=run_path + '/', property_string='accuracy')
    plot_property(main_run, run_cfg_list, ['model'], save_path=run_path + '/', property_string='accuracy')
    plot_property(main_run, run_cfg_list, ['param_drop_ratio'], save_path=run_path + '/', property_string='accuracy')

    main_run = RunConfig(run_path, property_string='data_upload')
    run_paths = os.listdir(path_prefix)
    run_paths = list(filter(lambda x: x.startswith('run_'), run_paths))
    run_cfg_list = [RunConfig(path_prefix + path, property_string='data_upload')
                for path in run_paths if RunConfig(path_prefix + path, property_string='data_upload').cfg_hash != main_run.cfg_hash]
    run_cfg_list = list(filter(lambda x: x._config is not None, run_cfg_list))

    plot_property(main_run, run_cfg_list, ['session_tag'], save_path=run_path + '/', property_string='data_upload')
    plot_property(main_run, run_cfg_list, ['algorithm'], save_path=run_path + '/', property_string='data_upload')
    plot_property(main_run, run_cfg_list, ['model'], save_path=run_path + '/', property_string='data_upload')

    main_run = RunConfig(run_path, property_string='learning_rate')  
    plot_property(main_run, [], ['session_tag'], save_path=run_path + '/', property_string='learning_rate')

    main_run = RunConfig(run_path, property_string='training_loss')
    run_paths = os.listdir(path_prefix)
    run_paths = list(filter(lambda x: x.startswith('run_'), run_paths))
    run_cfg_list = [RunConfig(path_prefix + path, property_string='training_loss')
                for path in run_paths if RunConfig(path_prefix + path, property_string='training_loss').cfg_hash != main_run.cfg_hash]
    run_cfg_list = list(filter(lambda x: x._config is not None, run_cfg_list))

    plot_property(main_run, run_cfg_list, ['session_tag'], save_path=run_path + '/', property_string='training_loss')
    plot_property(main_run, run_cfg_list, ['algorithm'], save_path=run_path + '/', property_string='training_loss')
    plot_property(main_run, run_cfg_list, ['model'], save_path=run_path + '/', property_string='training_loss')

    main_run = RunConfig(run_path, property_string='max_accuracy_over_data')
    run_paths = os.listdir(path_prefix)
    run_paths = list(filter(lambda x: x.startswith('run_'), run_paths))
    run_cfg_list = [RunConfig(path_prefix + path, property_string='max_accuracy_over_data')
                for path in run_paths if RunConfig(path_prefix + path, property_string='max_accuracy_over_data').cfg_hash != main_run.cfg_hash]
    run_cfg_list = list(filter(lambda x: x._config is not None, run_cfg_list))

    plot_property(main_run, run_cfg_list, ['session_tag'], save_path=run_path + '/', property_string='max_accuracy_over_data')
    plot_property(main_run, run_cfg_list, ['algorithm'], save_path=run_path + '/', property_string='max_accuracy_over_data')
    plot_property(main_run, run_cfg_list, ['model'], save_path=run_path + '/', property_string='max_accuracy_over_data')
    plot_property(main_run, run_cfg_list, ['param_drop_ratio'], save_path=run_path + '/', property_string='max_accuracy_over_data')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(prog='Plotting utility tool.')
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    path_prefix = args.path
    run_paths = os.listdir(path_prefix)
    run_paths = list(filter(lambda x: x.startswith('run_'), run_paths))
    run_cfg_list = [RunConfig(path_prefix + path) for path in run_paths]
    run_cfg_list = list(filter(lambda x: x._config is not None, run_cfg_list))

    import multiprocessing
    import tqdm

    tasks = [run.path_to_folder for run in run_cfg_list]

    with multiprocessing.Pool(processes=6) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(plot_config, tasks), total=len(tasks)):
            pass
