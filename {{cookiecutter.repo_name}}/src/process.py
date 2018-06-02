"""
Create processed files
"""
import pandas as pd
from exp_utils import from_dataframe_cache


@from_dataframe_cache('proc_train')
def get_train(config, force=False, **kwargs):
    tr = pd.read_csv(config['files']['raw_train'])
    return tr


@from_dataframe_cache('proc_test')
def get_test(config, force=False, **kwargs):
    te = pd.read_csv(config['files']['raw_test'])
    return te


process_funcs = {
    "proc_train": get_train,
    "proc_test": get_test,
}
