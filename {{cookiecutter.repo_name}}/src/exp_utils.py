"""Experiment Setup Utils"""
import datetime
import logging
import csv
import os

import pandas as pd
from sacred.observers.base import RunObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


LOGGING_FORMAT = '%(levelname)s: %(message)s'
TIME_FORMAT = "r%Y%m%d_%H%M%S"


def get_log_file_handler(log_fn, level=logging.INFO):
    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    return file_handler


class CSVObserver(RunObserver):

    COLS = ["run_id", "delta_time", "train", "valid"]

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):
        self.results_fn = 'artifacts/results.csv'
        self.run_id = ex_info['name']
        self.start_time = start_time
        self.record_local = config['record_local']

        if os.path.exists(self.results_fn):
            return
        with open(self.results_fn, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLS, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()

    def completed_event(self, stop_time, result):
        if not result or len(result) != 2 or not self.record_local:
            return
        d_time = (stop_time - self.start_time).total_seconds()
        result = {"run_id": self.run_id, "delta_time": f"{d_time:.2f}",
                  "train": result[1], "valid": result[0]}
        with open(self.results_fn, 'r') as f:
            df = pd.read_csv(f, index_col="run_id")

        new_row = pd.Series(result)
        df.loc[self.run_id] = new_row
        with open(self.results_fn, 'w') as f:
            df.to_csv(f)


def add_common_config(exp, record_local=True):
    exp.add_config(
        run_id=None,
        record_local=record_local,
    )
    exp.observers.append(CSVObserver())
    exp.captured_out_filter = apply_backspaces_and_linefeeds


def generate_run_dir(run_id):

    run_id = run_id or datetime.datetime.utcnow().strftime(TIME_FORMAT)
    run_dir = os.path.join('artifacts', run_id)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    return run_dir


def setup_run_dir_predict(run_id, config, log):

    run_dir = os.path.join('artifacts', run_id)
    if not run_id or not os.path.exists(run_dir):
        raise EnvironmentError(f"run_id needs to be set to predict")
    config['run_dir'] = run_dir

    # setup file logging
    log_fn = os.path.join(run_dir, 'log_predict.txt')
    file_hander = get_log_file_handler(log_fn)
    log.addHandler(file_hander)

    return run_dir


def setup_run_dir_train(run_id, config, log):

    run_dir = generate_run_dir(run_id)
    config['run_dir'] = run_dir

    # setup file logging
    log_fn = os.path.join(run_dir, 'log_train.txt')
    file_hander = get_log_file_handler(log_fn)
    log.addHandler(file_hander)

    return run_dir


def addMongoDBFromENV(exp):
    mongodb_url = os.environ.get("MONGODB_URL")
    mongodb_name = os.environ.get("MONGODB_NAME")

    if mongodb_url and mongodb_name:
        exp.append(MongoObserver.create(
            url=mongodb_url,
            db_name=mongodb_name
        ))
