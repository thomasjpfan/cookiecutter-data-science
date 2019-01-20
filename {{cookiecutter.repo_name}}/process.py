"""
Create processed files
"""
import fire

import pandas as pd

from runner import get_config
from utils import from_dataframe_cache


@from_dataframe_cache("files__proc_train")
def get_train(params, force=False, **kwargs):
    tr = pd.read_csv(params.files__raw_train)
    return tr


@from_dataframe_cache("files__proc_test")
def get_test(params, force=False, **kwargs):
    te = pd.read_csv(params.files__raw_test)
    return te


PROCESS_FUNCS = {"files__proc_train": get_train, "files__proc_test": get_test}

PROCESS_CHOICES = [
    key.split("files__proc_", maxsplit=1)[1] for key in PROCESS_FUNCS
]


def process(key, force=False):

    config = get_config()
    key = f"files__proc_{key}"
    try:
        process_fn = config[key]
    except KeyError:
        print(f"files/processed/{key} does not exists in config")
        return

    if process_fn.exists() and not force:
        print(f"{process_fn} already exists, use --force")
        return

    print(f"Processing {key} filename: {process_fn}")
    PROCESS_FUNCS[key](config, force)


if __name__ == "__main__":
    fire.Fire(process)
