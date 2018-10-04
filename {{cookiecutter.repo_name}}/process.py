"""
Create processed files
"""
import argparse

import pandas as pd

from utils import get_params, from_dataframe_cache


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


def process(args):
    key = args.key
    force = args.force

    params = get_params()

    key = f"files__proc_{key}"
    try:
        process_fn = params[key]
    except KeyError:
        print(f"files/processed/{key} does not exists in params")
        return

    if process_fn.exists() and not force:
        print(f"{process_fn} already exists, use --force")
        return

    print(f"Processing {key} filename: {process_fn}")
    PROCESS_FUNCS[key](params, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "key", choices=PROCESS_CHOICES, help="key of processed file"
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="rerun processing for key"
    )
    parser.set_defaults(func=process)

    args = parser.parse_args()
    args.func(args)
