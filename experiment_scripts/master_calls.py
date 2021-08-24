import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import sklearn.model_selection

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_directory)
from utils import RF, RFSH, IMP0, \
    ASKL_FULL, ASKL_FULL_RANDOM, ASKL_ITER, ASKL_ITER_RANDOM
from utils import dataset_dc, method_dc

this_directory = os.path.abspath(os.path.dirname(__file__))


NSEEDS = 10
tl_settings = {
    "1MIN": {
        "prtl": 10,
        "tl": 60,
        "memory": 4000,
    },
    "10MIN": {
        "prtl": 60,
        "tl": 600,
        "memory": 4000,
    },
    "20MIN": {
        "prtl": 120,
        "tl": 1200,
        "memory": 4000,
    },
    "60MIN": {
        "prtl": 360,
        "tl": 3600,
        "memory": 4000,
    },
    "10H": {
        "prtl": 3600,
        "tl": 36000,
        "memory": 4000,
    }
}

def write_cmd(cmd_list, cmd_file):
    if len(cmd_list) < 10000:
        with open(cmd_file, "w") as fh:
            fh.write("\n".join(cmd_list))
        print("Written %d commands to %s" % (len(cmd_list), cmd_file))
    else:
        print("Found more than 10000 cmds (%d)" % len(cmd_list))
        ct = 0
        while ct < len(cmd_list):
            ct += 10000
            with open(cmd_file + "_%d" % ct, "w") as fh:
                fh.write("\n".join(cmd_list[ct-10000:ct]))
            print("Written %d commands to %s" % (len(cmd_list[ct-10000:ct]), cmd_file))


def run_autosklearn(taskset, setting, methods, working_directory, nseeds, metric,
                    keep_predictions=False, initial_configurations_via_metalearning=0,
                    metadata_directory=None, autobuild_ensembles=False, max_mem_usage_models=None,
                    ignore_existing_output_dir=False, runcount_limit=None):
    # Creates cmd file to run autosklearn
    cmd_tmpl = "python %s/run_autosklearn/run_autosklearn.py --time-limit %d --memory-limit %d " \
               "--per-run-time-limit %d --working-directory %s --metric %s " \
               "--initial-configurations-via-metalearning %d --task-id %d -s %d"

    if autobuild_ensembles:
        cmd_tmpl += " --posthoc-ensemble"
    if max_mem_usage_models:
        cmd_tmpl += " --max-mem-usage-models %f" % max_mem_usage_models
    if ignore_existing_output_dir:
        cmd_tmpl += " --ignore-existing-output-dir"

    cmd_list = []
    for seed in range(nseeds):
        for tid in dataset_dc[taskset]:
            for set in methods:
                for method in set:
                    cmd_base = cmd_tmpl % (this_directory,
                                           tl_settings[setting]["tl"],
                                           tl_settings[setting]["memory"],
                                           tl_settings[setting]["prtl"],
                                           working_directory,
                                           metric,
                                           initial_configurations_via_metalearning,
                                           tid,
                                           seed)
                    for k in method_dc[method]:
                        cmd_base += " --%s %s" % (k, method_dc[method][k])
                    if keep_predictions:
                        cmd_base += " --keep-predictions"
                    if metadata_directory:
                        cmd_base += (" --metadata-directory %s" % metadata_directory)
                    if runcount_limit:
                        cmd_base += (" --runcount-limit %s" % runcount_limit)
                    cmd_list.append(cmd_base)

    if os.path.isdir(working_directory):
        print("Working directory %s already exists: Abort!" % working_directory)
        sys.exit(1)
    else:
        os.makedirs(working_directory)

    cmd_file = os.path.join(working_directory, "commands.txt")
    write_cmd(cmd_list, cmd_file)


def run_get_portfolio_configurations(taskset, input_dir, output_dir, methods, nseeds, setting,
                                     hostname, test_setting, metric):
    cmd_tmpl = "python %s/portfolio_matrix/get_portfolio_configurations.py --input_dir %s " \
               "--method %s --output_dir %s --nseeds %d --taskset %s"
    server_cmd_tmpl = 'python %s/portfolio_matrix/server.py --input-dir %s --host %s ' \
                      '--server-file %s --taskset %s'
    worker_cmd_tmpl = 'python %s/portfolio_matrix/worker.py --input-dir %s --memory-limit %s ' \
                      '--time-limit 10000 --per-run-time-limit %s --working-directory %s/matrix/ ' \
                      ' --server-file %s --metric %s --taskset %s'
    if test_setting:
        server_cmd_tmpl = server_cmd_tmpl + ' --test --wait-for 60'
        worker_cmd_tmpl = worker_cmd_tmpl + ' --test'

    cmd_list = []
    worker_cmd_dc = {}
    for set in methods:
        for idx, method in enumerate(set):
            cmd = cmd_tmpl % (this_directory, input_dir, method, output_dir, nseeds, taskset)
            cmd_list.append(cmd)

            worker_cmd_dc[method] = []

            # build server command
            server_file = os.path.join(output_dir, method, 'server.json')
            cmd = server_cmd_tmpl % (this_directory, output_dir + "/" + method, hostname,
                                     server_file, taskset)
            for k in ["searchspace", "evaluation", "iterative-fit", "early-stopping"]:
                cmd += " --%s %s" % (k, method_dc[method][k])
            if method_dc[method]["evaluation"] == "CV":
                cmd += " --cv %s" % method_dc[method]["cv"]
            worker_cmd_dc[method].append(cmd)

            # build worker command
            cmd = worker_cmd_tmpl % (this_directory, output_dir + "/" + method,
                                     tl_settings[setting]["memory"],
                                     tl_settings[setting]["prtl"], output_dir + "/" + method,
                                     server_file, metric, taskset)
            for i in range(500):
                worker_cmd_dc[method].append(cmd)

    if os.path.isdir(output_dir):
        print("Working directory %s already exists: Abort!" % output_dir)
        sys.exit(1)
    else:
        os.makedirs(output_dir)

    cmd_file = os.path.join(output_dir, "get_portfolio_configurations.cmd")
    with open(cmd_file, "w") as fh:
        fh.write("\n".join(cmd_list))
    print("Written %d commands to %s" % (len(cmd_list), cmd_file))

    for m in worker_cmd_dc:
        cmd_file = os.path.join(output_dir, "worker_%s.txt" % m)
        with open(cmd_file, "w") as fh:
            fh.write("\n".join(worker_cmd_dc[m]))
        print("Written %d commands to %s" % (len(worker_cmd_dc[m]), cmd_file))


def run_create_matrix(taskset, input_dir, output_dir, methods):
    cmd_tmpl = (
        "python %s/portfolio_matrix/create_matrix.py "
        "--working-directory %s/%s/matrix/ --save-to %s/%s/ --taskset %s"
    )
    cmds = []
    for set in methods:
        for method in set:
            cmd = cmd_tmpl % (this_directory, input_dir, method, input_dir, method, taskset)
            cmds.append(cmd)
    cmd_file = os.path.join(output_dir, "build_matrix.cmd")
    with open(cmd_file, "w") as fh:
        fh.write("\n".join(cmds))
    print("Written %d commands to %s" % (len(cmds), cmd_file))


def run_create_symlinks(input_dir):
    print(os.listdir(input_dir))
    replacements = {'_None_': '_SH-eta4-i_'}
    for dir in os.listdir(input_dir):
        for idx, method in enumerate(RF):
            for target, replace in replacements.items():
                if method == dir:
                    target_dir = os.path.join(input_dir, dir.replace(target, replace))
                    print("Create symlink from %s to %s" % (dir, target_dir))
                    os.symlink(dir, target_dir)


def run_create_portfolio(taskset, input_dir, output_dir, nseeds, portfolio_size, methods, setting,
                         test_setting):
    call_template = (
        'python %s/portfolio/build_portfolio.py '
        '--input-directory %s '
        '--portfolio-size %d '
    )
    call_template += ('--taskset %s ' % taskset)
    if test_setting:
        call_template = call_template + ' --test'

    commands = []
    commands_cv = []
    execute_portfolio_calls = []

    os.makedirs(output_dir, exist_ok=True)

    for set in methods:
        for idx, method in enumerate(set):
            for seed in range(nseeds):
                call = copy.copy(call_template)
                input_dir_ = os.path.join(input_dir, method)
                output_dir_ = os.path.join(output_dir, method)
                os.makedirs(output_dir_, exist_ok=True)
                portfolio_execution_output_directory = os.path.join(output_dir_, 'portfolio_execution')
                os.makedirs(portfolio_execution_output_directory, exist_ok=True)
                fidelities = method_dc[method]['fidelity']

                task_to_portfolio = {}
                splits = {}

                call = call + (' --fidelities %s' % fidelities)
                call = call + (' --seed %d' % seed)

                output_file = os.path.join(
                    output_dir_,
                    '%d_%s_%d.json' % (portfolio_size, fidelities, seed)
                )

                main_call = call + (' --output-file %s' % output_file)
                main_call = main_call % (this_directory, input_dir_, portfolio_size)
                commands.append(main_call)
                metadata_ids = np.array(dataset_dc[taskset])

                kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
                for split_ixd, (training_indices, test_indices) in enumerate(
                        kfold.split(metadata_ids)):
                    training_task_ids = metadata_ids[training_indices]
                    test_task_ids = metadata_ids[test_indices]
                    splits[split_ixd] = {}
                    splits[split_ixd]['train'] = [int(tidx) for tidx in training_task_ids]
                    splits[split_ixd]['test'] = [int(tidx) for tidx in test_task_ids]

                    output_file = os.path.join(
                        output_dir_,
                        '%d_%s_%d_%d.json' % (
                        portfolio_size, fidelities, seed, split_ixd)
                    )

                    tmp_call = call + ' --training-task-ids '
                    tmp_call = tmp_call + ' '.join([str(tti) for tti in training_task_ids])
                    tmp_call = tmp_call + ' --output-file %s' % output_file
                    tmp_call = tmp_call % (this_directory, input_dir_, portfolio_size)
                    commands_cv.append(tmp_call)

                    for task_id in test_task_ids:

                        task_to_portfolio[int(task_id)] = output_file

                        portfolio_execution_output_file = os.path.join(
                            portfolio_execution_output_directory,
                            '%d_%s_%d_%d.json' % (
                            portfolio_size, fidelities, seed, task_id)
                        )

                        execute_portfolio_call = (
                            'python %s/portfolio/execute_portfolio.py '
                            '--portfolio-file %s '
                            '--input-directory %s '
                            '--output-file %s '
                            '--fidelities %s '
                            '--taskset %s '
                            '--task-id %d '
                            '--max-runtime %d'
                        ) % (this_directory, output_file, input_dir_,
                            portfolio_execution_output_file,
                            fidelities, taskset, int(task_id),
                            tl_settings[setting]["tl"]
                        )
                        if test_setting:
                            execute_portfolio_call = execute_portfolio_call + ' --test'
                        execute_portfolio_calls.append(execute_portfolio_call)

                task_to_portfolio_filename = os.path.join(
                    output_dir_,
                    'task_to_portfolio_%d.json' % seed,
                )
                with open(task_to_portfolio_filename, 'wt') as fh:
                    json.dump(task_to_portfolio, fh, indent=4)
                splits_filename = os.path.join(
                    output_dir_,
                    'splits_%d.json' % seed,
                )
                with open(splits_filename, 'wt') as fh:
                    json.dump(splits, fh, indent=4)

    call_file = os.path.join(output_dir, 'build_portfolio.cmd')
    write_cmd(commands, call_file)

    cv_call_file = os.path.join(output_dir, 'build_portfolio_cv.cmd')
    write_cmd(commands_cv, cv_call_file)

    execute_portfolio_call_file = os.path.join(output_dir, 'execute_portfolio_cv.cmd')
    write_cmd(execute_portfolio_calls, execute_portfolio_call_file)


def run_autosklearn_with_portfolio(taskset, setting, methods, working_directory, nseeds, metric,
                                   portfolio_directory, keep_predictions=False,
                                   autobuild_ensembles=False,
                                   portfolio_from_dictionary_file=False,
                                   ignore_existing_output_dir=False,
                                   max_mem_usage_models=False,
                                   ):
    # Creates cmd file to run autosklearn
    cmd_tmpl = "python %s/run_autosklearn/run_autosklearn.py --time-limit %d --memory-limit %d " \
               "--working-directory %s --metric %s " \
               "--initial-configurations-via-metalearning %d --task-id %d -s %d"

    if autobuild_ensembles:
        cmd_tmpl += " --posthoc-ensemble"
    if max_mem_usage_models:
        cmd_tmpl += " --max-mem-usage-models %f" % max_mem_usage_models
    if ignore_existing_output_dir:
        cmd_tmpl += " --ignore-existing-output-dir"

    fixed_cmd_list = []
    for seed in range(nseeds):
        for tid in dataset_dc[taskset]:
            for set in methods:
                for method in set:
                    dir = os.path.join(working_directory)
                    cmd_base = cmd_tmpl % (this_directory,
                                           tl_settings[setting]["tl"],
                                           tl_settings[setting]["memory"],
                                           dir,
                                           metric,
                                           0,
                                           tid,
                                           seed)
                    for k in method_dc[method]:
                        cmd_base += " --%s %s" % (k, method_dc[method][k])
                    if keep_predictions:
                        cmd_base += " --keep-predictions"
                    fidelities = method_dc[method]['fidelity']

                    if portfolio_from_dictionary_file:
                        dictionary_file = os.path.join(
                            os.path.abspath(portfolio_directory),
                            method,
                            'task_to_portfolio_%d.json' % seed
                        )
                        with open(dictionary_file) as fh:
                            portfolio_dict = json.load(fh)
                        portfolio_file_name = portfolio_dict[str(tid)]
                    else:
                        portfolio_file_name = os.path.join(
                            os.path.abspath(portfolio_directory),
                            method,
                            "32_%s_%d.json" % (fidelities, seed),
                        )
                    cmd_base = cmd_base + " --portfolio-file " + portfolio_file_name
                    fixed_cmd_list.append(cmd_base)

    if os.path.isdir(working_directory):
        print("Working directory %s already exists: Abort!" % working_directory)
        sys.exit(1)
    else:
        os.makedirs(working_directory)

    cmd_file = os.path.join(working_directory, "commands.txt")
    write_cmd(fixed_cmd_list, cmd_file)


def run_AutoAuto_build(taskset, methods, portfolio_dir, normalization_dir, output_dir, nseeds,
                       splits_dir, metadata_type='portfolio', test=False):
    if os.path.isdir(output_dir):
        print("Working directory %s already exists: Abort!" % output_dir)
        sys.exit(1)
    else:
        os.makedirs(output_dir)
    metafeatures_directory = os.path.join(output_dir, 'metafeatures')
    os.makedirs(metafeatures_directory)

    commands = []

    call_template = 'python %s/autoauto/run_autoauto.py' % this_directory

    for selector_type in ('static', 'dynamic'):
        call = call_template + (' --type %s' % selector_type)
        call = call + (' --metadata-type %s' % metadata_type)
        call = call + (' --metafeatures-directory %s' % metafeatures_directory)
        call = call + (' --taskset %s' % taskset)
        if test:
            call = call + ' --test'

        call = call + (' --train-normalization-dir %s' % normalization_dir)
        call = call + (' --train-input-dir %s' % portfolio_dir)
        call = call + ' --only'
        for set in methods:
            for method in set:
                call = call + (' %s' % method)

        for seed in range(nseeds):
            call_with_seed = call + (' --seed %d' % seed)
            output_file = os.path.join(output_dir, selector_type, '%d.pkl' % seed)
            call_with_seed = call_with_seed + (' --output-file %s' % output_file)
            splits_file = glob.glob(os.path.join(splits_dir, '*', 'splits_%s.json' % seed))[0]
            call_with_seed = call_with_seed + (' --splits-file %s' % splits_file)
            commands.append(call_with_seed)

    cmd_file = os.path.join(output_dir, "commands.txt")
    write_cmd(cmd_list=commands, cmd_file=cmd_file)


def run_AutoAuto_simulate(taskset, selector_dir, run_with_portfolio_dir, output_dir, nseeds,
                          setting, add_symlinks=False, add_symlinks_and_stats_file=False,
                          add_no_fallback=False):

    if add_symlinks:
        assert os.path.isdir(output_dir)
    else:
        if os.path.isdir(output_dir):
            print("Working directory %s already exists: Skip!" % output_dir)
            # sys.exit(1)
        else:
            os.makedirs(output_dir)

    cmd_list = []
    call_template = 'python %s/run_autosklearn/simulate_autoautoml.py' % this_directory

    selector_mappinf = {
        'static': 'static',
        'static-no-fallback': 'static',
        'dynamic': 'dynamic',
        'dynamic-no-fallback': 'dynamic',
    }
    for selector_type in ('static', 'dynamic', 'dynamic-no-fallback', 'static-no-fallback'):

        if not add_no_fallback and selector_type in ('dynamic-no-fallback', 'static-no-fallback'):
            continue

        output_dir_ = os.path.join(output_dir, selector_type)
        input_dir = os.path.join(run_with_portfolio_dir, 'RF')

        for seed in range(nseeds):
            for task_id in dataset_dc[taskset]:
                selector_file = os.path.join(
                    selector_dir,
                    selector_mappinf[selector_type],
                    '%d.pkl' % seed
                )
                call = call_template + (' --selector-file %s' % selector_file)
                call = call + (' --task-id %d' % task_id)
                call = call + (' --seed %d' % seed)
                call = call + (' --input-dir %s' % input_dir)
                call = call + (' --output-dir %s' % output_dir_)
                call = call + (' --max-runtime-limit %s' % tl_settings[setting]["tl"])
                if add_symlinks:
                    call = call + ' --create-symlink --only-check-stats-file'
                elif add_symlinks_and_stats_file:
                    call = call + ' --create-symlink'
                if selector_type in ('dynamic-no-fallback', 'static-no-fallback'):
                    call = call + ' --disable-fallback'
                cmd_list.append(call)

    if not add_symlinks:
        cmd_file = os.path.join(output_dir, "commands.txt")
    else:
        cmd_file = os.path.join(output_dir, "commands_symlinks.txt")

    write_cmd(cmd_list, cmd_file)


def prune_run_with_portfolio(setting, commands_dir, autoauto_dir, rq_prefix):
    for prefix, cmd_file in [
        ('', os.path.join(commands_dir, '%s_cmds.txt' % setting)),
        ('fixed_', os.path.join(commands_dir, 'fixed_%s_cmds.txt' % setting)),
        ('learned_', os.path.join(commands_dir, 'learned_%s_cmds.txt' % setting)),
    ]:
        new_commands = []
        if not os.path.exists(cmd_file):
            continue
        with open(cmd_file) as fh:
            commands = fh.read().split('\n')
        glob_cmd = os.path.join(autoauto_dir, '*', '*', '*.json')
        globs = glob.glob(glob_cmd)
        print(glob_cmd, len(globs))
        for entry in globs:
            with open(entry) as fh:
                jason = json.load(fh)
            try:
                per_run_time_limit = jason['max_runtime_limit']
            except KeyError:
                continue
            if per_run_time_limit == 'None':
                runtime_limit = tl_settings[setting]['tl']
            else:
                runtime_limit = 'None'
            chosen_method = jason['chosen_method']
            seed = jason['seed']
            task_id = jason['task_id']
            portfolio_file_string = os.path.join(setting, 'ASKL_create_portfolio', chosen_method)
            fidelities = 'SH' if 'SH' in chosen_method else 'None'
            portfolio_file_string = os.path.join(
                portfolio_file_string,
                '32_%s_%s_%d.json' % (fidelities, runtime_limit, seed)
            )
            for command in commands:
                # The weird construct of the portfolio file string removes the time horizon indicator
                if '/'.join(portfolio_file_string.split('/')[1:]) in command and ('--task-id %d' % task_id) in command:
                    new_commands.append(command)

        new_commands_file = os.path.join(
            commands_dir,
            '%s%s%s_cmds_selected.txt' % (rq_prefix, prefix, setting),
        )
        write_cmd(new_commands, new_commands_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do',
                        choices=("ASKL_metadata",
                                 "ASKL_metadata_full",
                                 "ASKL_metadata_full_run_with_portfolio",
                                 "ASKL_improvement_zero",
                                 "ASKL_automldata_w_ensemble",
                                 "ASKL_automldata_w_ensemble_w_knd",
                                 "ASKL_automldata_run_with_portfolio_w_ensemble",
                                 "ASKL_automldata_baseline_iter",
                                 "ASKL_automldata_baseline_iter_no_metalearning",
                                 "ASKL_automldata_baseline_iter_random",
                                 "ASKL_automldata_baseline_full",
                                 "ASKL_automldata_baseline_full_no_metalearning",
                                 "ASKL_automldata_baseline_full_random",
                                 "ASKL_getportfolio",
                                 "run_create_matrix",
                                 "run_create_symlinks",
                                 "ASKL_create_portfolio",
                                 "AutoAuto_build",
                                 "AutoAuto_simulate",
                                 "AutoAuto_simulate_create_posthoc_symlinks",
                                 "prune_run_with_portfolio",
                                 "AutoAuto_build_full_data",
                                 "AutoAuto_simulate_full_data",
                                 "RQ1_AutoAuto_build",
                                 "RQ1_AutoAuto_simulate",
                                 "RQ1_AutoAuto_simulate_create_posthoc_symlinks",
                                 "RQ1_prune_run_with_portfolio",
                                 "RQ2.1_AutoAuto_build",
                                 "RQ2.1_AutoAuto_simulate",
                                 "RQ2.2_AutoAuto_build",
                                 "RQ2.2_AutoAuto_simulate",
                                 "RQ2.3_AutoAuto_build",
                                 "RQ2.3_AutoAuto_simulate",
                                 "RQ3.1_ASKL_run_with_portfolio_w_ensemble",
                                 "RQ3.1_AutoAuto_simulate",
                                 "RQ3.1_prune_run_with_portfolio"),
                        required=True)
    parser.add_argument('--setting', choices=list(tl_settings.keys()), required=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--v2', action='store_true')

    parser.add_argument('--hostname', type=str)
    parser.add_argument('--metric', default='balanced_accuracy',
                        choices=('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'r2',
                                 'mean_squared_error', 'root_mean_squared_error',
                                 'mean_absolute_error'
                                 )
                        )
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--max-mem-usage-models', type=float, default=None)
    parser.add_argument('--ignore-existing-output-dir', action='store_true')
    args = parser.parse_args()
    test_setting = args.test
    v2 = args.v2
    if v2 and test_setting:
        raise ValueError()
    metric = args.metric
    if test_setting:
        if metric == 'roc_auc':
            training_tasks = 'automl_test_metadata_binary'
            test_tasks = 'automl_test_benchmark_two_class'
        else:
            training_tasks = "automl_test_metadata"
            test_tasks = 'automl_test_benchmark'
    elif args.v2:
        if metric == 'roc_auc':
            training_tasks = 'automl_metadata2_binary'
            test_tasks = None
        else:
            training_tasks = "automl_metadata2"
            test_tasks = None
    else:
        if metric == 'roc_auc':
            training_tasks = 'automl_metadata_binary'
            test_tasks = None
        else:
            training_tasks = "automl_metadata"
            test_tasks = "openml_automl_benchmark"
    n_seeds = args.n_seeds
    max_mem_usage_models = args.max_mem_usage_models
    ignore_existing_output_dir = args.ignore_existing_output_dir

    if args.do == "ASKL_improvement_zero":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        run_autosklearn(taskset=test_tasks, setting=args.setting, methods=(IMP0, ),
                        working_directory=working_directory, nseeds=n_seeds, metric=metric,
                        max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_metadata":
        # This is clearly a subset of the runs conducted for ASKL_metadata_full, nevertheless,
        # we keep it separate as this constructs the data to build the portfolios,
        # while ASKL_metadata_full constructs data to build selectors on.
        # Also, this one here is limited to 5k samples!
        if n_seeds != 10:
            raise ValueError(
                'Changing the number of repetitions is not supported for %s!' % args.do
            )
        working_directory = os.path.join(this_directory, args.setting, args.do)
        run_autosklearn(taskset=training_tasks, setting=args.setting, methods=(RF, ),
                        working_directory=working_directory, nseeds=3, metric=metric,
                        max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir,
                        runcount_limit=5000)
    elif args.do == "ASKL_metadata_full":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        run_autosklearn(taskset=training_tasks, setting=args.setting, methods=(RF, RFSH),
                        working_directory=working_directory, nseeds=n_seeds,
                        autobuild_ensembles=True, metric=metric,
                        max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_metadata_full_run_with_portfolio":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        portfolio_directory = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")
        run_autosklearn_with_portfolio(taskset=training_tasks, setting=args.setting,
                                       methods=(RF, RFSH),
                                       working_directory=working_directory, nseeds=n_seeds,
                                       portfolio_directory=portfolio_directory,
                                       autobuild_ensembles=True,
                                       portfolio_from_dictionary_file=True,
                                       metric=metric,
                                       ignore_existing_output_dir=ignore_existing_output_dir,
                                       max_mem_usage_models=max_mem_usage_models)
    elif args.do == "ASKL_automldata_w_ensemble":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        run_autosklearn(taskset=test_tasks, setting=args.setting, methods=(RF, RFSH),
                        working_directory=working_directory, nseeds=n_seeds, autobuild_ensembles=True,
                        metric=metric, max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_automldata_w_ensemble_w_knd":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        metadata_dir = os.path.join(this_directory, 'kND/metadata/files/')
        run_autosklearn(taskset=test_tasks, setting=args.setting, methods=(RF, RFSH),
                        working_directory=working_directory, nseeds=n_seeds,
                        initial_configurations_via_metalearning=25,
                        metadata_directory=metadata_dir, metric=metric,
                        max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir,
                        autobuild_ensembles=True)
    elif args.do == "ASKL_automldata_baseline_iter":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        metadata_dir = os.path.join(this_directory, 'kND/metadata/files/')
        run_autosklearn(taskset=test_tasks, setting=args.setting,
                        methods=(ASKL_ITER, ), working_directory=working_directory, nseeds=n_seeds,
                        autobuild_ensembles=True, initial_configurations_via_metalearning=25,
                        metadata_directory=metadata_dir,
                        metric=metric, max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_automldata_baseline_iter_no_metalearning":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        metadata_dir = os.path.join(this_directory, 'kND/metadata/files/')
        run_autosklearn(taskset=test_tasks, setting=args.setting,
                        methods=(ASKL_ITER, ), working_directory=working_directory, nseeds=n_seeds,
                        autobuild_ensembles=True, initial_configurations_via_metalearning=0,
                        metadata_directory=metadata_dir,
                        metric=metric, max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_automldata_baseline_iter_random":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        metadata_dir = os.path.join(this_directory, 'kND/metadata/files/')
        run_autosklearn(taskset=test_tasks, setting=args.setting,
                        methods=(ASKL_ITER_RANDOM, ), working_directory=working_directory,
                        nseeds=n_seeds, autobuild_ensembles=True, metric=metric,
                        max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir,
                        metadata_directory=metadata_dir,
                        initial_configurations_via_metalearning=0)
    elif args.do == "ASKL_automldata_baseline_full":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        run_autosklearn(taskset=test_tasks, setting=args.setting,
                        methods=(ASKL_FULL, ), working_directory=working_directory, nseeds=n_seeds,
                        autobuild_ensembles=True, initial_configurations_via_metalearning=25,
                        metric=metric, max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_automldata_baseline_full_no_metalearning":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        run_autosklearn(taskset=test_tasks, setting=args.setting,
                        methods=(ASKL_FULL, ), working_directory=working_directory, nseeds=n_seeds,
                        autobuild_ensembles=True, initial_configurations_via_metalearning=0,
                        metric=metric, max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_automldata_baseline_full_random":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        run_autosklearn(taskset=test_tasks, setting=args.setting,
                        methods=(ASKL_FULL_RANDOM, ), working_directory=working_directory,
                        nseeds=n_seeds, autobuild_ensembles=True, metric=metric,
                        max_mem_usage_models=max_mem_usage_models,
                        ignore_existing_output_dir=ignore_existing_output_dir)
    elif args.do == "ASKL_getportfolio":
        input_dir = os.path.join(this_directory, args.setting, "ASKL_metadata")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        hostname = args.hostname
        if hostname is None:
            raise ValueError('hostname must not be None for task ASKL_getportfolio')
        elif n_seeds != 10:
            raise ValueError(
                'Changing the number of repetitions is not supported for %s!' % args.do
            )
        run_get_portfolio_configurations(taskset=training_tasks, input_dir=input_dir,
                                         output_dir=output_dir, methods=(RF, ), nseeds=3,
                                         setting=args.setting, hostname=hostname,
                                         test_setting=test_setting, metric=metric)
    elif args.do == "run_create_matrix":
        input_dir = os.path.join(this_directory, args.setting, 'ASKL_getportfolio')
        output_dir = os.path.join(this_directory, args.setting, 'ASKL_getportfolio')
        run_create_matrix(
            taskset=training_tasks, input_dir=input_dir, output_dir=output_dir, methods=(RF, )
        )
    elif args.do == "run_create_symlinks":
        input_dir = os.path.join(this_directory, args.setting, 'ASKL_getportfolio')
        run_create_symlinks(input_dir)
    elif args.do == "ASKL_create_portfolio":
        input_dir = os.path.join(this_directory, args.setting, 'ASKL_getportfolio')
        output_dir = os.path.join(this_directory, args.setting, args.do)
        run_create_portfolio(taskset=training_tasks, input_dir=input_dir, output_dir=output_dir,
                             nseeds=n_seeds, portfolio_size=32, methods=(RF, RFSH),
                             setting=args.setting, test_setting=test_setting)
    elif args.do == "ASKL_automldata_run_with_portfolio_w_ensemble":
        working_directory = os.path.join(this_directory, args.setting, args.do)
        portfolio_directory = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")
        run_autosklearn_with_portfolio(taskset=test_tasks, setting=args.setting,
                                       methods=(RF, RFSH), working_directory=working_directory,
                                       nseeds=n_seeds, portfolio_directory=portfolio_directory,
                                       autobuild_ensembles=True, metric=metric)
    elif args.do == "AutoAuto_build":
        normalization_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full_run_with_portfolio")
        splits_dir = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")
        portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full_run_with_portfolio", "RF")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        run_AutoAuto_build(taskset=training_tasks, methods=(RF, RFSH), nseeds=n_seeds,
                           portfolio_dir=portfolio_dir, normalization_dir=normalization_dir,
                           output_dir=output_dir, metadata_type='full_runs', test=test_setting,
                           splits_dir=splits_dir)
    elif args.do == "AutoAuto_simulate":
        selector_dir = os.path.join(this_directory, args.setting, "AutoAuto_build")
        run_with_portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_automldata_run_with_portfolio_w_ensemble",)
        output_dir = os.path.join(this_directory, args.setting, args.do)
        run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
                              add_symlinks_and_stats_file=True,
                              add_no_fallback=True)
    # elif args.do == "AutoAuto_simulate_create_posthoc_symlinks":
    #     selector_dir = os.path.join(this_directory, args.setting, "AutoAuto_build")
    #     run_with_portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_run_with_portfolio")
    #     output_dir = os.path.join(this_directory, args.setting, 'AutoAuto_simulate')
    #     run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
    #                           run_with_portfolio_dir=run_with_portfolio_dir,
    #                           output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
    #                           add_symlinks=True)
    # elif args.do == "prune_run_with_portfolio":
    #     commands_dir = os.path.join(this_directory, args.setting, "ASKL_run_with_portfolio")
    #     autoauto_dir = os.path.join(this_directory, args.setting, 'AutoAuto_simulate')
    #     prune_run_with_portfolio(setting=args.setting, commands_dir=commands_dir,
    #                              autoauto_dir=autoauto_dir, rq_prefix='')
    elif args.do == "RQ1_AutoAuto_build":
        normalization_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full_run_with_portfolio")
        portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full_run_with_portfolio", "RF")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        splits_dir = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")
        run_AutoAuto_build(taskset=training_tasks, methods=(RF, RFSH), nseeds=n_seeds,
                           portfolio_dir=portfolio_dir, normalization_dir=normalization_dir,
                           output_dir=output_dir, metadata_type='full_runs_ensemble',
                           test=test_setting, splits_dir=splits_dir)
    elif args.do == "RQ1_AutoAuto_simulate":
        selector_dir = os.path.join(this_directory, args.setting, "RQ1_AutoAuto_build")
        run_with_portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_automldata_run_with_portfolio_w_ensemble")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
                              add_symlinks_and_stats_file=True,
                              add_no_fallback=True)
    # elif args.do == "RQ1_AutoAuto_simulate_create_posthoc_symlinks":
    #     selector_dir = os.path.join(this_directory, args.setting, "AutoAuto_build")
    #     run_with_portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_automldata_run_with_portfolio_w_ensemble")
    #     output_dir = os.path.join(this_directory, args.setting, 'RQ1_AutoAuto_simulate')
    #     run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
    #                           run_with_portfolio_dir=run_with_portfolio_dir,
    #                           output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
    #                           add_symlinks=True)
    # elif args.do == "RQ1_prune_run_with_portfolio":
    #     commands_dir = os.path.join(this_directory, args.setting, "ASKL_automldata_run_with_portfolio_w_ensemble")
    #     autoauto_dir = os.path.join(this_directory, args.setting, 'RQ1_AutoAuto_simulate')
    #     prune_run_with_portfolio(setting=args.setting, commands_dir=commands_dir,
    #                              autoauto_dir=autoauto_dir, rq_prefix='RQ1_')
    elif args.do == "RQ2.1_AutoAuto_build":
        normalization_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full_run_with_portfolio")
        portfolio_dir = os.path.join(this_directory, args.setting,
                                     "ASKL_metadata_full_run_with_portfolio",
                                     "RF",)
        output_dir = os.path.join(this_directory, args.setting, args.do)
        splits_dir = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")

        # No successive halving
        output_dir_ = os.path.join(this_directory, output_dir, 'no_sh')
        run_AutoAuto_build(taskset=training_tasks, methods=(RF, ), nseeds=n_seeds,
                           portfolio_dir=portfolio_dir, normalization_dir=normalization_dir,
                           output_dir=output_dir_, metadata_type='full_runs_ensemble',
                           test=test_setting, splits_dir=splits_dir)

        # Only successive halving
        output_dir_ = os.path.join(this_directory, output_dir, 'only_sh')
        run_AutoAuto_build(taskset=training_tasks, methods=(RFSH, ), nseeds=n_seeds,
                           portfolio_dir=portfolio_dir, normalization_dir=normalization_dir,
                           output_dir=output_dir_, metadata_type='full_runs_ensemble',
                           test=test_setting, splits_dir=splits_dir)

        # No cross-validation
        output_dir_ = os.path.join(this_directory, output_dir, 'no_cv')
        run_AutoAuto_build(taskset=training_tasks,
                           methods=(("RF_None_holdout_iterative_es_if",
                                     "RF_SH-eta4-i_holdout_iterative_es_if"), ),
                           nseeds=n_seeds, portfolio_dir=portfolio_dir,
                           normalization_dir=normalization_dir, output_dir=output_dir_,
                           metadata_type='full_runs_ensemble',
                           test=test_setting, splits_dir=splits_dir)

        # only cross-validation
        output_dir_ = os.path.join(this_directory, output_dir, 'only_cv')
        run_AutoAuto_build(taskset=training_tasks,
                           methods=(("RF_None_3CV_iterative_es_if",
                                     "RF_None_5CV_iterative_es_if",
                                     "RF_None_10CV_iterative_es_if",
                                     "RF_SH-eta4-i_3CV_iterative_es_if",
                                     "RF_SH-eta4-i_5CV_iterative_es_if",
                                     "RF_SH-eta4-i_10CV_iterative_es_if"),),
                           nseeds=n_seeds, portfolio_dir=portfolio_dir,
                           normalization_dir=normalization_dir, output_dir=output_dir_,
                           metadata_type='full_runs_ensemble',
                           test=test_setting, splits_dir=splits_dir)
    elif args.do == "RQ2.1_AutoAuto_simulate":
        for subset in ('no_sh', 'only_sh', 'no_cv', 'only_cv'):
            selector_dir = os.path.join(this_directory, args.setting, "RQ2.1_AutoAuto_build", subset)
            run_with_portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_automldata_run_with_portfolio_w_ensemble")
            output_dir = os.path.join(this_directory, args.setting, args.do, subset)
            run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
                                  run_with_portfolio_dir=run_with_portfolio_dir,
                                  output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
                                  add_symlinks_and_stats_file=True, add_no_fallback=True)
    elif args.do == "RQ2.2_AutoAuto_build":
        normalization_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full_run_with_portfolio")
        portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full", "RF")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        splits_dir = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")
        run_AutoAuto_build(taskset=training_tasks, methods=(RF, RFSH), nseeds=n_seeds,
                           portfolio_dir=portfolio_dir, normalization_dir=normalization_dir,
                           output_dir=output_dir, metadata_type='full_runs_ensemble',
                           test=test_setting, splits_dir=splits_dir)
    elif args.do == "RQ2.2_AutoAuto_simulate":
        selector_dir = os.path.join(this_directory, args.setting, "RQ2.2_AutoAuto_build")
        run_with_portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_automldata_w_ensemble")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
                              add_symlinks_and_stats_file=True, add_no_fallback=True)
    elif args.do == "RQ2.3_AutoAuto_build":
        normalization_dir = os.path.join(this_directory, args.setting, "ASKL_metadata_full_run_with_portfolio")
        portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        splits_dir = os.path.join(this_directory, args.setting, "ASKL_create_portfolio")
        run_AutoAuto_build(taskset=training_tasks, methods=(RF, RFSH), nseeds=n_seeds,
                           portfolio_dir=portfolio_dir, normalization_dir=normalization_dir,
                           output_dir=output_dir, metadata_type='portfolio',
                           test=test_setting, splits_dir=splits_dir)
    elif args.do == "RQ2.3_AutoAuto_simulate":
        selector_dir = os.path.join(this_directory, args.setting, "RQ2.3_AutoAuto_build")
        run_with_portfolio_dir = os.path.join(this_directory, args.setting, "ASKL_automldata_run_with_portfolio_w_ensemble")
        output_dir = os.path.join(this_directory, args.setting, args.do)
        run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
                              run_with_portfolio_dir=run_with_portfolio_dir,
                              output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
                              add_symlinks_and_stats_file=True,
                              add_no_fallback=True)

    # elif args.do == "RQ3.1_ASKL_run_with_portfolio_w_ensemble":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #         working_directory = os.path.join(this_directory, args.setting, args.do, setting)
    #         portfolio_directory = os.path.join(this_directory, setting, "ASKL_create_portfolio")
    #         run_autosklearn_with_portfolio(taskset=test_tasks, setting=args.setting,
    #                                        methods=(RF, RFSH), working_directory=working_directory,
    #                                        nseeds=n_seeds, portfolio_directory=portfolio_directory,
    #                                        autobuild_ensembles=True)
    # elif args.do == "RQ3.1_AutoAuto_simulate":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #         selector_dir = os.path.join(this_directory, args.setting, "AutoAuto_build")
    #         run_with_portfolio_dir = os.path.join(this_directory, args.setting,
    #                                               "RQ3.1_ASKL_run_with_portfolio_w_ensemble",
    #                                               setting)
    #         output_dir = os.path.join(this_directory, args.setting, args.do, setting)
    #         run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
    #                               run_with_portfolio_dir=run_with_portfolio_dir,
    #                               output_dir=output_dir, nseeds=n_seeds, setting=args.setting)
    # elif args.do == "RQ3.1_prune_run_with_portfolio":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #         commands_dir = os.path.join(this_directory, args.setting,
    #                                     "RQ3.1_ASKL_run_with_portfolio_w_ensemble", setting)
    #         autoauto_dir = os.path.join(this_directory, args.setting, 'RQ3.1_AutoAuto_simulate', setting)
    #         prune_run_with_portfolio(setting=args.setting, commands_dir=commands_dir,
    #                                  autoauto_dir=autoauto_dir, rq_prefix='')
    # elif args.do == "RQ3.1_AutoAuto_simulate_create_posthoc_symlinks":
    #     for setting in tl_settings:
    #         if args.setting == setting:
    #             continue
    #
    #         selector_dir = os.path.join(this_directory, args.setting, "AutoAuto_build")
    #         run_with_portfolio_dir = os.path.join(this_directory, args.setting,
    #                                               "RQ3.1_ASKL_run_with_portfolio_w_ensemble",
    #                                               setting)
    #         output_dir = os.path.join(this_directory, args.setting, 'RQ3.1_AutoAuto_simulate', setting)
    #         run_AutoAuto_simulate(taskset=test_tasks, selector_dir=selector_dir,
    #                               run_with_portfolio_dir=run_with_portfolio_dir,
    #                               output_dir=output_dir, nseeds=n_seeds, setting=args.setting,
    #                               add_symlinks=True)
    else:
        print("Don't know what to do: Exit!")

