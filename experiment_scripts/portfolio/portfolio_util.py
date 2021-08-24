from typing import Dict, List

import numpy as np
import pandas as pd


def reformat_data(
    matrix: Dict[int, pd.DataFrame],
    task_ids: List[int],
    configurations: List[str],
):
    lc_length = 1
    for task_id in task_ids:
        for config_id in configurations:
            try:
                if 'test_learning_curve' in matrix[task_id].loc[config_id]['additional_run_info']:
                    tmp_lc_length = len(matrix[task_id].loc[config_id][
                        'additional_run_info'][
                        'test_learning_curve'])
                    lc_length = max(lc_length, tmp_lc_length)
            except KeyError as e:
                print(
                    task_id, type(task_id), config_id, type(config_id),
                    matrix[task_id].index, matrix[task_id].columns,
                )
                raise e

    print('Maximal learning curve length', lc_length)
    y_valid = np.ones((len(task_ids), len(configurations), lc_length)) * np.NaN
    y_test = np.ones(y_valid.shape) * np.NaN
    runtimes = np.ones(y_valid.shape) * np.NaN
    config_id_to_idx = dict()
    task_id_to_idx = dict()
    for j, task_id in enumerate(task_ids):
        for k, config_id in enumerate(configurations):

            task_id_to_idx[task_id] = j
            config_id_to_idx[config_id] = k

            mmtc = matrix[task_id].loc[config_id]
            if 'error' in mmtc['additional_run_info']:
                y_valid[j][k][:] = 1.0
                y_test[j][k][:] = 1.0
                runtimes[j][k][:] = mmtc['runtime']
            elif 'test_learning_curve' in mmtc['additional_run_info']:
                lc_length = len(mmtc['additional_run_info']['learning_curve'])
                y_valid[j][k][:lc_length] = mmtc['additional_run_info']['learning_curve']
                y_test[j][k][:lc_length] = mmtc['additional_run_info']['test_learning_curve']
                runtimes[j][k][:lc_length] = mmtc['additional_run_info'][
                    'learning_curve_runtime']
                if lc_length < y_valid.shape[2]:
                    y_valid[j][k][lc_length:] = y_valid[j, k, -1]
                    y_test[j][k][lc_length:] = y_test[j, k, -1]
                    runtimes[j][k][lc_length:] = runtimes[j, k, -1]
            else:
                y_valid[j][k][0] = mmtc['loss']
                y_test[j][k][0] = mmtc['additional_run_info']['test_loss']
                runtimes[j][k][0] = mmtc['runtime']

    return y_valid, y_test, runtimes, config_id_to_idx, task_id_to_idx


def normalize_matrix(matrix):
    normalized_matrix = matrix.copy()
    minima = np.nanmin(np.nanmin(normalized_matrix, axis=2), axis=1)
    maxima = np.nanmax(np.nanmax(normalized_matrix, axis=2), axis=1)
    diff = maxima - minima
    diff[diff == 0] = 1
    for task_idx in range(normalized_matrix.shape[0]):
        normalized_matrix[task_idx] = (
                (normalized_matrix[task_idx] - minima[task_idx]) / diff[task_idx]
        )

    assert (
        np.all((normalized_matrix >= 0) | (~np.isfinite(normalized_matrix)))
        and np.all((normalized_matrix <= 1) | (~np.isfinite(normalized_matrix)))
    ), (
        normalized_matrix, (normalized_matrix >= 0) | (~np.isfinite(normalized_matrix))
    )
    return normalized_matrix
