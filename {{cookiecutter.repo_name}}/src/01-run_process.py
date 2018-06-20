"""
Create processed files
"""
import argparse

from exp_utils import get_params
from process import process_funcs


def process_key(params, key, force):
    if not key.startswith("files__proc_"):
        print(f"{key} is not process key in neptune.yaml")
        return

    try:
        process_fn = params[key]
    except KeyError:
        print(f'files/processed/{key} does not exists in params')
        return

    if process_fn.exists() and not force:
        print(f'{process_fn} already exists, use --force')
        return

    print(f'Processing {key} filename: {process_fn}')
    process_funcs[key](params, force)


def process(key, force):
    params = get_params()

    if key:
        process_key(params, key, force)
        return

    for key in process_funcs:
        process_key(params, key, force)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-k', '--key', type=str)
    parser.add_argument('-f', '--force', action='store_true')

    args = parser.parse_args()

    process(**vars(args))
