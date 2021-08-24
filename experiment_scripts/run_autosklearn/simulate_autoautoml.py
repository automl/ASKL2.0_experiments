import argparse
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

import sys
this_dir = os.path.dirname(__file__)
main_dir = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.append(main_dir)
autoautodir = os.path.abspath(os.path.join(this_dir, '..', 'autoauto'))
sys.path.append(autoautodir)
from utils import load_task, get_meta_features, automl_metadata, compute_meta_features


parser = argparse.ArgumentParser()
parser.add_argument('--selector-file', type=str, required=True)
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument('-s', '--seed', type=int, required=True)
parser.add_argument('--input-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--create-symlink', action='store_true')
parser.add_argument('--only-check-stats-file', action='store_true')
parser.add_argument('--max-runtime-limit', required=True)
parser.add_argument('--disable-fallback', action='store_true')
args = parser.parse_args()

selector_file = args.selector_file
seed = args.seed
task_id = args.task_id
input_dir = args.input_dir
output_dir = args.output_dir
create_symlink = args.create_symlink
only_check_stats_file = args.only_check_stats_file
max_runtime_limit = args.max_runtime_limit
disable_fallback = args.disable_fallback

with open(selector_file, 'rb') as fh:
    selector = pickle.load(fh)

methods_to_choose_from = selector.strategies_
if disable_fallback:
    if hasattr(selector, 'selector'):
        selector = selector.selector

X_train, y_train, _, _, _ = load_task(task_id)
min_num_samples_per_class = np.min(np.unique(y_train, return_counts=True)[1])

meta_features = compute_meta_features(X_train, y_train)
del X_train
del y_train
meta_features = pd.DataFrame({task_id: meta_features}).transpose()
meta_features = meta_features[['NumberOfFeatures', 'NumberOfInstances']]

print(selector.__dict__)
if hasattr(selector, 'selector'):
    print(selector.selector.__dict__)
prediction = selector.predict(meta_features).loc[task_id]
for mtcf, pred in zip(methods_to_choose_from, prediction):
    print(mtcf, pred)

if np.sum(prediction) == 0:
    raise ValueError('Cannot choose from empty array')
assert len(prediction.shape) == 1
assert prediction.shape[0] == len(methods_to_choose_from)
method = prediction.idxmax()

os.makedirs(output_dir, exist_ok=True)

stats_file_name = os.path.join(output_dir, 'stats_%s_%s.json' % (task_id, seed))
if only_check_stats_file:
    with open(stats_file_name) as fh:
        additional_info = json.load(fh)
    assert input_dir == additional_info['input_dir']
    assert set([float(pred) for pred in prediction]) == set(additional_info['predictions'])
    assert set(list(methods_to_choose_from)) == set(additional_info['methods_to_choose_from'])
    assert method == additional_info['chosen_method']
    assert task_id == additional_info['task_id']
    assert seed == additional_info['seed']
    assert max_runtime_limit == additional_info['max_runtime_limit']
else:
    additional_info = dict()
    additional_info['input_dir'] = input_dir
    additional_info['predictions'] = [float(pred) for pred in prediction]
    additional_info['methods_to_choose_from'] = list(methods_to_choose_from)
    additional_info['chosen_method'] = method
    additional_info['task_id'] = task_id
    additional_info['seed'] = seed
    additional_info['max_runtime_limit'] = max_runtime_limit
    with open(stats_file_name, 'wt') as fh:
        json.dump(additional_info, fh, indent=4)

if create_symlink:
    n_dirs = len(output_dir.split('/')) - output_dir.split('/').index('experiment_scripts')
    target = ['..'] * (n_dirs - 1)
    target.append(input_dir[re.search('experiment_scripts', input_dir).end() + 1: ])
    target.append('%s_%s_%s_0_0' % (method, task_id, seed))
    target = '/'.join(target)
    source = os.path.join(output_dir, 'autoauto_%s_%s' % (task_id, seed))
    print('Created symlink from %s to %s!' % (source, target), flush=True)
    os.symlink(target, source)
