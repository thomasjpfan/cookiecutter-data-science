"""
Create processed files
"""
import pandas as pd
from exp_utils import from_dataframe_cache


@from_dataframe_cache('files__proc_train')
def get_train(params, force=False, **kwargs):
    tr = pd.read_csv(params.files__raw_train)
    return tr


@from_dataframe_cache('files__proc_test')
def get_test(params, force=False, **kwargs):
    te = pd.read_csv(params.files__raw_test)
    return te


process_funcs = {
    "files__proc_train": get_train,
    "files__proc_test": get_test,
}
