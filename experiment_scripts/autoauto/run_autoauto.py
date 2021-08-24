import argparse
import collections
import glob
import json
import os
import pickle
import sys

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_directory)
sys.path.append(os.path.abspath(os.path.join(this_directory, '..')))
sys.path.append(os.path.abspath(os.path.join(this_directory, '.')))

import numpy as np
import pandas as pd
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from smac.tae import StatusType
from smac.runhistory.runhistory import RunValue
from smac.intensification.simple_intensifier import SimpleIntensifier

import utils
import autoauto.policy_selectors
import autoauto_experiments.metrics
import autoauto.autoauto_utils

pd.set_option('display.width', 999)
pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)

default_strategies = [
    'RF_SH-eta4-i_holdout_iterative_es_if',
    "RF_None_holdout_iterative_es_if",
    "RF_SH-eta4-i_3CV_iterative_es_if",
    "RF_None_3CV_iterative_es_if",
    "RF_SH-eta4-i_5CV_iterative_es_if",
    "RF_None_5CV_iterative_es_if",
    "RF_SH-eta4-i_10CV_iterative_es_if",
    "RF_None_10CV_iterative_es_if"
]


parser = argparse.ArgumentParser()
parser.add_argument('--metadata-type',
                    choices=['portfolio', 'full_runs', 'full_runs_ensemble'],
                    default='portfolio')
parser.add_argument('--metafeatures-directory', type=str, required=True)
parser.add_argument('--train-input-dir', type=str, required=True)
parser.add_argument('--train-normalization-dir', type=str, required=True)
parser.add_argument('--output-file', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--taskset', type=str, required=True)
parser.add_argument('--splits-file', type=str, required=True)
parser.add_argument('--type', choices=['static', 'dynamic'])
parser.add_argument('--test', action='store_true')
parser.add_argument('--only', nargs='*', type=str)

args = parser.parse_args()
metadata_type = args.metadata_type
metafeatures_directory = args.metafeatures_directory
train_input_dir = args.train_input_dir
train_normalization_dir = args.train_normalization_dir
output_file = args.output_file
seed = args.seed
taskset = args.taskset
splits_file = args.splits_file
rng = np.random.RandomState(seed)
autoauto_type = args.type
test = args.test
only = args.only

with open(splits_file) as fh:
    splits = json.load(fh)


if only:
    default_strategies = [ds for ds in default_strategies if ds in only]


X_train, y_train = autoauto.autoauto_utils.load_X_and_y(
    utils.dataset_dc[taskset],
    metafeatures_directory,
    train_input_dir,
    seed,
    metadata_type,
    allow_non_exist=False,
    average_runs=False,
    metafeatures_to_use=['NumberOfFeatures', 'NumberOfInstances'],
    only=only,
)

(
    min_diff_dc_train,
    minima_for_methods_train,
    maxima_for_methods_train,
    minima_for_tasks_train,
    maxima_for_tasks_train,
) = (
    utils.get_normalization_constants(
        train_normalization_dir,
        task_ids=utils.dataset_dc[taskset],
        load=False,
        n_seeds=1 if test else 10,
    )
)

regret = autoauto_experiments.metrics.get_regret(min_diff_dc_train)


print(X_train, y_train)


if autoauto_type == 'dynamic':

    cs = autoauto.policy_selectors.OVORF.get_configuration_space(X_train.shape[1])
    cs.seed(rng.randint(0, 1000))
    n_smac_iter = 10 if test else 200
    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": n_smac_iter,
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         "limit_resources": "false",
                         "output_dir": "none",
                         })
    smac_optimizer = SMAC4AC(
        scenario=scenario,
        rng=rng,
        tae_runner=lambda x: 0,
        intensifier=SimpleIntensifier,
    )
    initial_design_configs = smac_optimizer.solver.initial_design.select_configurations()

    best_loss = np.inf
    best_model = None
    best_oob_predictions = {}
    training_data = {}
    training_data['metafeatures'] = X_train.to_dict()
    training_data['columns'] = y_train.columns.to_list()
    training_data['y_values'] = [[float(_) for _ in row] for row in y_train.to_numpy()]
    training_data['minima_for_methods'] = minima_for_methods_train
    training_data['maxima_for_methods'] = maxima_for_methods_train
    training_data['tie_break_order'] = default_strategies

    for config_idx in range(n_smac_iter):
        print(config_idx, n_smac_iter)

        intent, run_info = smac_optimizer.solver.intensifier.get_next_run(
            challengers=initial_design_configs,
            incumbent=smac_optimizer.solver.incumbent,
            chooser=smac_optimizer.solver.epm_chooser,
            run_history=smac_optimizer.solver.runhistory,
            repeat_configs=smac_optimizer.solver.intensifier.repeat_configs,
            num_workers=1,
        )
        initial_design_configs = [c for c in initial_design_configs if c != run_info.config]
        smac_optimizer.solver.runhistory.add(
            config=run_info.config,
            cost=10000,
            time=0.0,
            status=StatusType.RUNNING,
            instance_id=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
        )
        run_info.config.config_id = smac_optimizer.solver.runhistory.config_ids[run_info.config]

        selectors = []
        y_true_tmp = pd.DataFrame()
        y_pred_tmp = pd.DataFrame()
        for split_idx, splits_for_idx in splits.items():
            train_tasks = splits_for_idx['train']
            test_tasks = splits_for_idx['test']

            selector = autoauto.policy_selectors.OVORF(
                configuration=run_info.config,
                random_state=rng,
                n_estimators=10 if test else 500,
                tie_break_order=default_strategies,
            )
            selector = autoauto.policy_selectors.FallbackWrapper(selector, default_strategies)
            selector.fit(
                X=X_train.loc[train_tasks],
                y=y_train.loc[train_tasks],
                minima=minima_for_methods_train,
                maxima=maxima_for_methods_train,
            )
            selectors.append(selector)
            prediction = selector.predict(X_train.loc[test_tasks])
            y_true_tmp = pd.concat((y_true_tmp, y_train.loc[test_tasks]))
            y_pred_tmp = pd.concat((y_pred_tmp, prediction))

        loss = regret(y_true_tmp, y_pred_tmp, nan_strategy='ignore')

        print(loss, best_loss)
        run_value = RunValue(cost=loss, time=1.0, status=StatusType.SUCCESS, starttime=1.0,
                             endtime=1.0, additional_info={})
        smac_optimizer.solver._incorporate_run_results(run_info, run_value, 10000)

        if loss < best_loss:
            print(run_info.config)

            selector = autoauto.policy_selectors.OVORF(
                configuration=run_info.config,
                random_state=rng,
                n_estimators=10 if test else 500,
                tie_break_order=default_strategies,
            )
            selector = autoauto.policy_selectors.FallbackWrapper(selector, default_strategies)
            selector.fit(
                X=X_train,
                y=y_train,
                minima=minima_for_methods_train,
                maxima=maxima_for_methods_train,
            )

            best_loss = loss
            best_model = selector
            best_oob_predictions = y_pred_tmp
            best_oob_targets = y_true_tmp

    regrets_rf = regret(y_true_tmp, best_oob_predictions, nan_strategy='ignore')
    training_data['configuration'] = best_model.selector.configuration

else:

    training_data = {}
    selector = autoauto.policy_selectors.SingleBest(loss_function=regret, random_state=rng)
    selector = autoauto.policy_selectors.FallbackWrapper(selector, default_strategies)
    selector.fit(
        X=X_train,
        y=y_train,
        minima=minima_for_methods_train,
        maxima=maxima_for_methods_train,
    )
    best_model = selector
    regrets_rf = None

y_true_tmp = pd.DataFrame()
y_pred_tmp = pd.DataFrame()
for split_idx, splits_for_idx in splits.items():
    train_tasks = splits_for_idx['train']
    test_tasks = splits_for_idx['test']

    selector = autoauto.policy_selectors.SingleBest(loss_function=regret, random_state=rng)
    selector = autoauto.policy_selectors.FallbackWrapper(selector, default_strategies)
    selector.fit(
        X=X_train.loc[train_tasks],
        y=y_train.loc[train_tasks],
        minima=minima_for_methods_train,
        maxima=maxima_for_methods_train,
    )
    prediction = selector.predict(X_train.loc[test_tasks])
    y_true_tmp = pd.concat((y_true_tmp, y_train.loc[test_tasks]))
    y_pred_tmp = pd.concat((y_pred_tmp, prediction))
regrets_single_best = regret(y_true_tmp, y_pred_tmp, nan_strategy='ignore')

regret_random = []
regret_oracle = []
base_method_regets = {method: [] for method in y_train.columns}
normalized_regret = y_train.copy()
# Normalize each column given the minimum and maximum ever observed on these tasks
for task_id in y_train.index:

    mask = np.isfinite(y_train.loc[task_id].to_numpy())
    if np.sum(mask) == 0:
        continue

    diff = maxima_for_tasks_train[task_id] - minima_for_tasks_train[task_id]
    if diff == 0:
        diff = 1

    normalized_regret.loc[task_id] = (
        normalized_regret.loc[task_id] - minima_for_tasks_train[task_id]
    ) / diff

    regret_random.extend([
        float(value)
        for value in np.random.choice(
            normalized_regret.loc[task_id][mask], size=1000, replace=True,
        )
    ])

    reg_oracle = float(np.nanmin(normalized_regret.loc[task_id]))

    if np.isfinite(reg_oracle):
        regret_oracle.append(reg_oracle)
    for method_idx, method in enumerate(y_train.columns):
        base_method_regets[method].append(normalized_regret.loc[task_id][method_idx])

full_oracle_perf = normalized_regret.min(axis=1).mean()
print('Oracle performance', full_oracle_perf)
for i in range(normalized_regret.shape[1]):
    subset_oracle_perf = normalized_regret.drop(normalized_regret.columns[i], axis=1).min(
        axis=1).mean()
    print(normalized_regret.columns[i], subset_oracle_perf - full_oracle_perf)

if regrets_rf:
    print('Regret rf', np.mean(regrets_rf))
print('Regret single best', np.mean(regrets_single_best))
print('Regret random', np.mean(regret_random))
print('Regret oracle', np.mean(regret_oracle))

output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

with open(output_file, 'wb') as fh:
    pickle.dump(best_model, fh)

with open(output_file + '.json', 'w') as fh:
    json.dump(training_data, fh, indent=4)
