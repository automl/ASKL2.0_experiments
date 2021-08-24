import argparse
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
from typing import List
import unittest.mock

from autosklearn.data import xy_data_manager
from autosklearn.evaluation import ExecuteTaFuncWithQueue, get_cost_of_crash
from autosklearn.constants import BINARY_CLASSIFICATION
import autosklearn.util.backend
from autosklearn.metrics import accuracy, balanced_accuracy, roc_auc, log_loss, r2, \
    mean_squared_error, mean_absolute_error, root_mean_squared_error, Scorer
from ConfigSpace import Configuration
from ConfigSpace.read_and_write.json import read
from autosklearn.automl import AutoML
import numpy as np
import openml
import requests
from smac.stats.stats import Stats
from smac.runhistory.runhistory import RunInfo

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_directory)
sys.path.append(os.path.abspath(os.path.join(this_directory, '..')))

from utils import dataset_dc
from utils import add_classifier_wo_early_stopping

openml.config.server = "https://www.openml.org/api/v1/xml"


def load_task(task_id, memory_limit, seed):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    train_idx, test_idx = task.get_train_test_split_indices()
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    dataset = openml.datasets.get_dataset(task.dataset_id)
    _, _, cat, _ = dataset.get_data(target=task.target_name)
    del _
    del dataset
    cat = ['categorical' if c else 'numerical' for c in cat]
    X_train, y_train = AutoML.subsample_if_too_large(
        X=X_train,
        y=y_train,
        logger=unittest.mock.Mock(),
        memory_limit=memory_limit,
        task=autosklearn.constants.MULTICLASS_CLASSIFICATION,
        seed=seed,
    )
    return X_train, y_train, X_test, y_test, cat


def create_datamanager(task_id, backend, memory_limit, seed):

    X_train, y_train, X_test, y_test, cat = load_task(task_id, memory_limit, seed)

    dm = xy_data_manager.XYDataManager(
        X=X_train, y=y_train,
        X_test=X_test, y_test=y_test,
        task=BINARY_CLASSIFICATION,
        feat_type=cat,
        dataset_name=str(task_id),
    )

    backend.save_datamanager(datamanager=dm)
    del dm


def run_configuration(backend, config_id, task_id, configuration,
                      run_args, memory_limit, per_run_time_limit, metric):
    evaluation, iterative_fit, early_stopping, N_FOLDS, searchspace = run_args

    # TODO make this an argument from the command line!
    scenario_mock = unittest.mock.Mock()
    scenario_mock.wallclock_limit = per_run_time_limit * 100
    scenario_mock.algo_runs_timelimit = per_run_time_limit * 100
    scenario_mock.ta_run_limit = np.inf
    stats = Stats(scenario_mock)
    stats.ta_runs = 2

    # Resampling strategies
    kwargs = {}
    if evaluation == "holdout" and iterative_fit:
        resampling_strategy = 'holdout-iterative-fit'
    elif evaluation == "holdout" and not iterative_fit:
        resampling_strategy = 'holdout'
    elif evaluation == "CV" and not iterative_fit:
        resampling_strategy = 'cv'
        kwargs = {'folds': N_FOLDS}
    elif evaluation == "CV" and iterative_fit:
        resampling_strategy = 'cv-iterative-fit'
        kwargs = {'folds': N_FOLDS}
    else:
        raise ValueError("Unknown resampling strategy", evaluation)

    iterative_wo_early_stopping = ['extra_trees', 'PassiveAggressiveWOEarlyStopping',
                                   'random_forest',
                                   'SGDWOEarlyStopping',
                                   'GradientBoostingClassifierWOEarlyStopping']
    iterative_w_early_stopping = ['extra_trees', 'passive_aggressive', 'random_forest', 'sgd',
                                  'gradient_boosting', 'mlp']

    if not early_stopping:
        add_classifier_wo_early_stopping()

    if searchspace == "iterative":
        include_estimator = iterative_w_early_stopping if early_stopping else iterative_wo_early_stopping
        include_preprocessor = ["no_preprocessing", ]
    elif searchspace == "full":
        assert early_stopping is True
        include_estimator = None
        include_preprocessor = None
    # elif searchspace == 'only-iterative-nopreproc':
    #    include_estimator = iterative_w_early_stopping if early_stopping else iterative_wo_early_stopping
    #    include_preprocessor = ["no_preprocessing", ]
    # elif searchspace == 'only-iterative-cheappreproc':
    #    include_estimator = iterative_w_early_stopping if early_stopping else iterative_wo_early_stopping
    #    include_preprocessor = ["no_preprocessing", 'kitchen_sinks', 'polynomial', 'select_percentile_classification', 'select_rates']
    # elif searchspace == 'only-iterative':
    #    include_estimator = iterative_w_early_stopping if early_stopping else iterative_wo_early_stopping
    #    include_preprocessor = None
    # elif searchspace == "gb":
    #    include_estimator = ['GradientBoostingClassifierWOEarlyStopping'] if early_stopping else ['GradientBoostingClassifierWEarlyStopping']
    #    include_preprocessor = None
    else:
        raise ValueError(searchspace)


    stats.start_timing()
    tae = ExecuteTaFuncWithQueue(
        backend=backend,
        autosklearn_seed=3,
        resampling_strategy=resampling_strategy,
        metric=metric,
        logger=logging.getLogger(name="%s_%s" % (task_id, config_id)),
        pynisher_context='forkserver',
        initial_num_run=2,
        stats=stats,
        runhistory=None,
        run_obj='quality',
        par_factor=1,
        all_scoring_functions=False,
        output_y_hat_optimization=True,
        include={"classifier": include_estimator,
                 "feature_preprocessor": include_preprocessor},
        exclude=None,
        memory_limit=memory_limit,
        disable_file_output=True,
        init_params=None,
        abort_on_first_run_crash=False,
        cost_for_crash=get_cost_of_crash(balanced_accuracy),
        port=None,
        **kwargs
    )
    configuration.config_id = 1

    # Finally run configuration
    _, runvalue = tae.run_wrapper(
        RunInfo(
            seed=1,
            config=configuration,
            instance=None,
            cutoff=per_run_time_limit,
            instance_specific=None,
            capped=False,
        )
    )

    status, cost, runtime, additional_run_info = runvalue.status, runvalue.cost, runvalue.time, runvalue.additional_info
    return status, cost, runtime, additional_run_info


def main(
    configurations_file: str,
    configurationspace_file: str,
    working_directory: str,
    memory_limit: int,
    time_limit: int,
    per_run_time_limit: int,
    host: str,
    port: int,
    task_ids: List[int],
    metric: Scorer,
):

    start_time = time.time()

    try:
        os.makedirs(working_directory)
    except:
        pass

    with open(configurationspace_file) as fh:
        configspace = read(fh.read())
    with open(configurations_file) as fh:
        configuration_dictionaries = json.load(fh)
    configurations = {}
    for i, entry in configuration_dictionaries.items():
        config = Configuration(configuration_space=configspace, values=entry)
        configurations[i] = config

    for task_id in list(np.random.permutation(task_ids)):
        print('Evaluating task', task_id)

        tmpdir = tempfile.mkdtemp()
        backend = None
        task_id = int(task_id)

        while True:

            if (time.time() - start_time) > (time_limit - per_run_time_limit - 30):
                print(
                    'Reached time limit! (%f > %f)' % (
                        (time.time() - start_time), (time_limit - per_run_time_limit - 30)
                    )
                )
                exit(0)

            # Connect to server, retry for some time
            for i in range(1, 101):
                try:
                    rval = requests.request(
                        'GET',
                        'http://%s:%d/%d' % (host, port, task_id,)
                    )
                    break
                except Exception as e:
                    if i < 10:
                        print('Could not establish connection due to', e, 'sleeping', i % 5 + 1, 'seconds')
                        time.sleep(i % 5 + 1)
                    else:
                        raise

            response_string = rval.content.decode('utf8')
            try:
                response = json.loads(response_string)
            except:
                print(response_string)
                print(rval)
                raise

            counter = response['counter']
            # resampling_strategy, iterative_fit, early_stopping, N_FOLDS, searchspace
            run_args = response['run_args']
            evaluation = run_args[0]
            print("Going to run count:", counter)
            if counter == -1:
                break

            for job_number, (config_id, configuration) in enumerate(
                list(sorted(list(configurations.items())))
            ):

                if job_number != counter:
                    continue

                print(
                    'Evaluating task %d, %s_%s_%s_%s, config %s (%d/%d)' % (
                        task_id,
                        run_args[4], run_args[0] if run_args[0] != "cv" else "%s%d" % (evaluation, run_args[3]),
                        "nif" if not run_args[1] else "if",
                        "nes" if not run_args[2] else "es",
                        config_id, job_number + 1, len(configurations)
                    )
                )

                output_dir = os.path.join(working_directory, str(task_id))
                output_path = os.path.join(
                    output_dir,
                    '%s_%s.json' % (run_args[0], str(config_id)),
                )

                # First check if it's necessary to do something more
                # complicated!
                # This should actually be done with a timeout...
                try:
                    with open(output_path) as fh:
                        json.load(fh)
                    print('Exists')
                    continue
                except:
                    pass

                try:
                    os.makedirs(output_dir)
                except:
                    pass

                try:
                    os.symlink(output_path, output_path + '.lock')

                    if backend is None:
                        print('Loading', task_id, 'running', config_id)
                        backend = autosklearn.util.backend.create(
                            temporary_directory=os.path.join(tmpdir, '%d_%d_%s' % (
                            task_id, job_number, evaluation)),
                            output_directory=None,
                            delete_tmp_folder_after_terminate=False,
                            delete_output_folder_after_terminate=True,
                        )
                        context = multiprocessing.get_context('spawn')
                        process = context.Process(
                            target=create_datamanager,
                            kwargs=dict(
                                task_id=task_id,
                                backend=backend,
                                memory_limit=memory_limit,
                                seed=1,
                            ),
                        )
                        process.start()
                        process.join()
                        if process.exitcode != 0:
                            raise ValueError(process.exitcode)
                    else:
                        print(
                            'Re-using loaded',
                            task_id,
                            'running',
                            config_id
                        )
                        pass

                    status, cost, runtime, additional_run_info = (
                        run_configuration(
                            backend,
                            config_id,
                            task_id,
                            configuration,
                            run_args,
                            memory_limit,
                            per_run_time_limit,
                            metric
                        )
                    )

                    with open(output_path, 'w') as fh:
                        json.dump({
                            'task_id': task_id,
                            'configuration_id': config_id,
                            'status': status.value,
                            'loss': cost,
                            'runtime': runtime,
                            'additional_run_info': additional_run_info,
                        }, fh, indent=4)
                except FileExistsError:
                    pass
                except Exception as e:
                    traceback.print_exc()
                    os.remove(output_path)
                    raise e
                finally:
                    delete_iter = 0
                    print(os.path.islink(output_path + '.lock'), output_path + '.lock')
                    while os.path.islink(output_path + '.lock'):
                        delete_iter += 1
                        try:
                            os.remove(output_path + '.lock')
                        except Exception as e:
                            print(e)
                            time.sleep(1)
                        if delete_iter > 10:
                            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to folder with incumbents.json, space.json and task_to_id.json.')
    parser.add_argument('--working-directory', type=str, required=True,
                        help='Directory which is used to store all temporary '
                             'files.')
    parser.add_argument('--memory-limit', type=int, required=True)
    parser.add_argument('--time-limit', type=int, required=True)
    parser.add_argument('--per-run-time-limit', type=int, required=True)
    parser.add_argument('--server-file', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--taskset', type=str, required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()


    configurations_file = os.path.join(args.input_dir, "incumbents.json")
    configurationspace_file = os.path.join(args.input_dir, "space.json")
    working_directory = args.working_directory
    memory_limit = args.memory_limit
    time_limit = args.time_limit
    per_run_time_limit = args.per_run_time_limit
    server_file = args.server_file
    task_ids = dataset_dc[args.taskset]

    host = None
    wait_for_seconds = 3600
    for i in range(wait_for_seconds):
        try:
            with open(server_file) as fh:
                jason = json.load(fh)
                host = jason['host']
                port = jason['port']
            break
        except Exception as e:
            time.sleep(1)
    if host is None:
        raise ValueError('Could not load server file after %d seconds' % wait_for_seconds)

    metric = args.metric
    metric = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc,
        'log_loss': log_loss,
        'r2': r2,
        'mean_squared_error': mean_squared_error,
        'root_mean_squared_error': root_mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
    }[metric]

    main(
        configurations_file,
        configurationspace_file,
        working_directory,
        memory_limit,
        time_limit,
        per_run_time_limit,
        host,
        port,
        task_ids,
        metric,
    )
    exit()

