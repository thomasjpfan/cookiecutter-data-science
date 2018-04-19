'''Experiment Setup Utils'''
from contextlib import suppress
from functools import wraps
import datetime
import logging
import csv
import os

import yaml
import pandas as pd
import numpy as np
from sacred.observers.base import RunObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


LOGGING_FORMAT = '> %(message)s'
TIME_FORMAT = '%Y%m%d_%H%M%S'
VAL_TRAIN_SCORE = "val_train_score.txt"


class CSVObserver(RunObserver):

    COLS = ['model_id', 'delta_time', 'train', 'valid']

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):
        if command != 'train':
            self.record_local = False
            return
        self.results_fn = 'artifacts/results.csv'
        self.model_id = config['model_id']
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
        result = {'model_id': self.model_id,
                  'delta_time': f'{d_time:.2f}',
                  'train': result[1], 'valid': result[0]}
        with open(self.results_fn, 'r') as f:
            df = pd.read_csv(f, index_col='model_id')

        new_row = pd.Series(result)
        df.loc[self.model_id] = new_row
        with open(self.results_fn, 'w') as f:
            df.to_csv(f)


class ArtifactObserver(RunObserver):

    def __init__(self, logger):
        self.logger = logger

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):
        run_dir = config['run_dir']

        if command == 'predict' and not os.path.exists(run_dir):
            raise EnvironmentError(f'run_id must exist to predict')

        os.makedirs(run_dir, exist_ok=True)
        log_fn = os.path.join(run_dir, f'log_{command}.txt')
        file_hander = get_log_file_handler(log_fn)

        self.logger.addHandler(file_hander)
        self.val_test_score_fn = os.path.join(run_dir, VAL_TRAIN_SCORE)

    def completed_event(self, stop_time, result):
        if not result or len(result) != 2:
            return
        val_train_score = np.array(result)
        np.savetxt(self.val_test_score_fn, val_train_score)


def get_config(root_dir="."):
    config_fn = os.path.join(root_dir, "config.yaml")
    with open(config_fn, "r") as f:
        config = yaml.load(f)

    if root_dir == ".":
        return config

    with suppress(KeyError):
        files = config['files']
        for folder_type, files_dict in files.items():
            for file_id, path in files_dict.items():
                config['files'][folder_type][file_id] = os.path.join(root_dir, path)

    return config


def add_common_config(exp, record_local=True):
    exp.add_config(
        run_id=datetime.datetime.utcnow().strftime(TIME_FORMAT),
        record_local=record_local,
        name=exp.path
    )
    exp.add_config("config.yaml")

    @exp.config
    def run_dir_config(name, run_id):
        model_id = f"{name}_{run_id}"  # noqa
        run_dir = os.path.join('artifacts', model_id)  # noqa

    exp.logger = get_stream_logger(exp.path)
    exp.observers.append(CSVObserver())
    exp.observers.append(ArtifactObserver(exp.logger))
    add_monogodb_from_env(exp.observers)
    add_pushover_handler_from_env(exp.logger)
    exp.captured_out_filter = apply_backspaces_and_linefeeds


def add_monogodb_from_env(exp):
    mongodb_url = os.environ.get('MONGODB_URL')
    mongodb_name = os.environ.get('MONGODB_NAME')

    if mongodb_url and mongodb_name:
        exp.append(MongoObserver.create(
            url=mongodb_url,
            db_name=mongodb_name
        ))


def add_pushover_handler_from_env(log):
    pushover_user_token = os.environ.get('NOTIFIERS_PUSHOVER_USER')
    pushover_token = os.environ.get('NOTIFIERS_PUSHOVER_TOKEN')

    if pushover_user_token and pushover_token:
        from notifiers.logging import NotificationHandler
        h = NotificationHandler('pushover')
        h.setLevel(logging.WARNING)
        log.addHandler(h)


def get_log_file_handler(log_fn, level=logging.INFO):
    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    return file_handler


def get_stream_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

    logger.addHandler(stream_handler)

    return logger


def from_cache(key):
    def cache_decor(f):
        @wraps(f)
        def wrapper(config, force):
            fn = config['files']['processed'][key]
            if os.path.exists(fn) and not force:
                return pd.read_parquet(fn)
            output = f(config, force)
            output.to_parquet(fn)
            return output
        return wrapper
    return cache_decor
