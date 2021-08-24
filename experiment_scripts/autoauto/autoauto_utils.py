import collections
import glob
import json
import os
import sys

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_directory)
sys.path.append(os.path.abspath(os.path.join(this_directory, '..')))
sys.path.append(os.path.abspath(os.path.join(this_directory, '.')))

import numpy as np
import pandas as pd

import utils


def load_X_and_y(task_ids, metafeatures_directory, input_directory, seed,
                 metadata_type, allow_non_exist, average_runs, metafeatures_to_use, only=None):
    meta_features = dict()
    for task_id in task_ids:
        meta_features[task_id] = utils.get_meta_features(task_id, metafeatures_directory)
    meta_features = pd.DataFrame(meta_features).transpose()
    meta_features = meta_features[metafeatures_to_use]
    meta_features.fillna(0, inplace=True)
    meta_features = meta_features.astype(float)

    methods_to_choose_from = dict()
    performance_matrix = collections.defaultdict(lambda: collections.defaultdict(list))

    if average_runs:
        seeds = list(range(10))
    else:
        seeds = (seed, )

    for seed_ in seeds:
        if metadata_type == 'portfolio':
            glop_path = os.path.join(
                input_directory, '*', 'portfolio_execution', '*_*_%d_*.json' % seed
            )
            print(glop_path)
            files = glob.glob(glop_path)
            for filepath in files:
                full_directory = os.path.split(filepath)[0]
                method_key = os.path.split(filepath)[0].split('/')[-2]
                filename = os.path.split(filepath)[1]
                filename = filename.replace('.json', '')
                filename = filename.split('_')
                path_template = os.path.join(
                    full_directory,
                    ('_'.join(filename[:3])) + '_%d.json'
                )
                portfolio_file = os.path.join(full_directory, '_'.join(filename[:3]) + '.json')
                portfolio_file = portfolio_file.replace('portfolio_execution/', '')
                methods_to_choose_from[method_key] = {'path_template': path_template,
                                                      'portfolio_file': portfolio_file, }
            print(methods_to_choose_from)

        elif metadata_type in ('full_runs', 'full_runs_ensemble'):
            if metadata_type == 'full_runs':
                glop_path = os.path.join(input_directory, '*_%d_0_0' % seed_, 'result.json')
            elif metadata_type == 'full_runs_ensemble':
                glop_path = os.path.join(
                    input_directory, '*_%d_0_0' % seed_,
                    'ensemble_results_*_*_%d_0.000000thresh_50size_1.000000best.json' % seed_
                )
            else:
                raise ValueError(metadata_type)

            files = glob.glob(glop_path)
            if len(files) == 0:
                print(glop_path)
                raise ValueError('Could not find a result file at %s' % glop_path)
            for filepath in files:
                method_key = os.path.split(filepath)[0].split('/')[-1]
                method_key = method_key.split('_')
                task_id = method_key[-4]
                method = '_'.join(method_key[:-4])
                path_template = os.path.join(
                    input_directory,
                    method + '_%d_' + str(method_key[-3]) + '_0_0',
                    os.path.split(filepath)[1].replace(str(task_id), '%d')
                )
                methods_to_choose_from[method] = {'path_template': path_template}

        else:
            raise ValueError(metadata_type)

        keys = list(methods_to_choose_from.keys())

        keys = sorted(keys)
        methods_to_choose_from = {key: methods_to_choose_from[key] for key in keys}

        # Second, load the cross-validated performance matrices

        not_found = 0
        backup = 0
        for method, additional_info in methods_to_choose_from.items():
            for task_id in task_ids:
                if metadata_type == 'full_runs':
                    path = additional_info['path_template'] % task_id
                    backup_path = None
                elif metadata_type == 'full_runs_ensemble':
                    path = additional_info['path_template'] % (task_id, task_id)
                    backup_path = os.path.join(os.path.split(path)[0], 'result.json')
                else:
                    path = additional_info['path_template'] % task_id
                    print(path)
                    backup_path = None
                if not os.path.exists(path) and (backup_path is None or not os.path.exists(backup_path)):
                    if allow_non_exist:
                        loss = np.NaN
                        not_found += 1
                    else:
                        raise ValueError(path)
                else:
                    if not os.path.exists(path):
                        path = backup_path
                        backup += 1
                    with open(path) as fh:
                        results = json.load(fh)
                        if '0' in results:
                            loss = results['0']['loss']
                        else:
                            loss = results['50']['loss']
                        loss = loss if np.isfinite(loss) else 1.0
                performance_matrix[method][task_id].append(loss)

    for method, additional_info in methods_to_choose_from.items():
        for task_id in task_ids:
            performance_matrix[method][task_id] = np.mean(performance_matrix[method][task_id])
    performance_matrix = pd.DataFrame(performance_matrix)
    if only:
        performance_matrix = performance_matrix[only]
    return meta_features, performance_matrix
