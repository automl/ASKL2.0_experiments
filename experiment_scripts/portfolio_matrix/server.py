import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from multiprocessing import Value
import os
import random
from socketserver import ThreadingMixIn
import socket
import sys
import threading
import time

this_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_directory)
sys.path.append(os.path.abspath(os.path.join(this_directory, '..')))

from utils import dataset_dc

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


class LastInteraction:

    def __init__(self):
        self.timestamp = Value('d', 0.0)

    def set(self, value):
        with self.timestamp.get_lock():
            self.timestamp.value = value

    def get(self):
        with self.timestamp.get_lock():
            value = self.timestamp.value
        return value


def check_and_shutdown(last_interaction: LastInteraction, server, wait_for: float):
    while True:
        current_time = time.time()
        if current_time - last_interaction.get() > wait_for:
            break
        time.sleep(10)
    server.shutdown()


def start_server(n_configurations, n_jobs, counters, host, server_file, run_args,
                 last_interaction, wait_for):
    handler = lambda *args: PostHandler(n_configurations, n_jobs, counters,
                                        run_args, last_interaction, *args)

    server = None
    exception = None
    port = None
    for _ in range(100):
        try:
            port = random.randint(10000, 50000)
            server = ThreadedHTTPServer((host, port), handler)
            break
        except Exception as e:
            exception = e
    if server is None:
        raise ValueError('Could not start the server due to %s' % exception)

    thread = threading.Thread(target=check_and_shutdown, args=(last_interaction, server, wait_for))
    thread.start()

    with open(server_file, 'w') as fh:
        json.dump({'port': port, 'host': host}, fh)

    server.serve_forever()

    os.remove(server_file)


class Counter:

    def __init__(self):
        self.counter = Value('i', 0)

    def get_value_and_increment(self, max_values):
        with self.counter.get_lock():
            value = self.counter.value
            if value >= max_values:
                value = -1
            else:
                self.counter.value += 1
        return value


class PostHandler(BaseHTTPRequestHandler):

    def __init__(self, n_configurations, n_jobs, counters, run_args, last_interaction, *args):
        # For reasons I don't understand the assignment has to happen before calling the parent
        # __init__.
        self.n_configurations = n_configurations
        self.run_args = run_args
        self.n_jobs = n_jobs
        self.counters = counters
        self.last_interaction = last_interaction

        BaseHTTPRequestHandler.__init__(self, *args)

    def do_POST(self):

        # Begin the response
        self.send_response(200)
        self.end_headers()

        # Very basic logging!
        print('Post')

        return

    def do_GET(self):
        path = self.path
        path = path.replace('/', '')

        # Begin the response
        self.send_response(200)
        self.end_headers()

        task_id = int(path)
        value = counters[task_id].get_value_and_increment(max_values=len(configurations))

        self.last_interaction.set(time.time())

        response = {
            'task_id': task_id,
            'run_args': self.run_args,
            'counter': value,
            'n_jobs': self.n_jobs,
        }
        response_string = json.dumps(response, indent=4)

        # Very basic logging!
        print('Get', task_id, value, flush=True)

        self.wfile.write(response_string.encode('utf8'))
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to folder with incumbents.json, space.json and task_to_id.json.')
    parser.add_argument('--searchspace', choices=("full", "iterative"), required=True)
    parser.add_argument('--evaluation', choices=("holdout", "CV"), required=True)
    parser.add_argument('--cv', choices=(3, 5, 10), type=int, required=False)  # depends
    parser.add_argument("--iterative-fit", choices=("True", "False"), required=True)
    parser.add_argument("--early-stopping", choices=("True", "False"), required=True)
    parser.add_argument('--taskset', type=str, required=True)
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--server-file', type=str, required=True)
    parser.add_argument('--wait-for', type=float, default=3600.0)
    args = parser.parse_args()

    configurations_file = os.path.join(args.input_dir, "incumbents.json")
    configurationspace_file = os.path.join(args.input_dir, "space.json")

    evaluation = args.evaluation
    iterative_fit = args.iterative_fit == "True"
    early_stopping = args.early_stopping == "True"
    if ("_nif" in args.input_dir and iterative_fit)\
            or ("_if" in args.input_dir and not iterative_fit):
        raise ValueError("Wrong early stopping!", args.input_dir, iterative_fit)
    if ("_nes" in args.input_dir and early_stopping)\
            or ("_es" in args.input_dir and not early_stopping):
        raise ValueError("Wrong iterative_fit!: ", args.input_dir, early_stopping)
    if args.searchspace not in args.input_dir:
        raise ValueError("Wrong searchspace!: ", args.input_dir, args.searchspace)

    host = args.host
    if host == 'rz.ki.privat':
        host = '%s.%s' % (socket.gethostname(), host)
    elif host == 'login1.nemo.privat':
        host = socket.gethostname()
    else:
        raise ValueError(
            "Unknown host %s, if you're reproducing the examples, please add your host" % host
        )
    server_file = args.server_file
    wait_for = args.wait_for

    task_ids = dataset_dc[args.taskset]

    with open(configurations_file) as fh:
       configurations = json.load(fh)

    n_jobs = len(configurations) * len(task_ids)
    counters = {task_id: Counter() for task_id in task_ids}

    print('Starting server for:')
    print('    %d configurations' % len(configurations))
    print('    %d tasks' % len(task_ids))
    print('    %d total' % n_jobs)

    last_interactino = LastInteraction()
    last_interactino.set(time.time())

    start_server(
        n_configurations=len(configurations),
        n_jobs=n_jobs,
        counters=counters,
        run_args=(evaluation, iterative_fit, early_stopping, args.cv, args.searchspace),
        host=host,
        server_file=server_file,
        last_interaction=last_interactino,
        wait_for=wait_for,
    )
