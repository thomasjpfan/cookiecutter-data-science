import os
import csv

import pandas as pd
import numpy as np
from sacred.observers.base import RunObserver

from .logging import get_log_file_handler


class CSVObserver(RunObserver):

    COLS = ['model_id', 'start_time', 'delta_time', 'train', 'valid']

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):
        if command not in ['train', 'train_hp']:
            self.record_local = False
            return
        self.results_fn = 'artifacts/results.csv'
        self.model_id = config['model_id'] + '_' + command
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
                  'start_time': self.start_time.isoformat(),
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
        self.val_test_score_fn = os.path.join(run_dir, "val_train_score.txt")

    def completed_event(self, stop_time, result):
        if not result or len(result) != 2:
            return
        val_train_score = np.array(result)
        np.savetxt(self.val_test_score_fn, val_train_score)
