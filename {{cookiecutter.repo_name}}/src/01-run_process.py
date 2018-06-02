"""
Create processed files
"""
import argparse

from exp_utils import get_config
from process import get_train, get_test

process_funcs = {
    "proc_train": get_train,
    "proc_test": get_test,
}


def process_key(config, key, force):
    if not key.startswith("proc_"):
        print(f"{key} is not process key in config.yaml")
        return

    try:
        process_fn = config['files'][key]
    except KeyError:
        print(f'files/processed/{key} does not exists in config')
        return

    if process_fn.exists() and not force:
        print(f'{process_fn} already exists, use --force')
        return

    print(f'Processing {key} filename: {process_fn}')
    process_funcs[key](config, force)


def process(key, force):
    config = get_config()

    if key:
        process_key(config, key, force)
        return

    for key in process_funcs:
        process_key(config, key, force)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-k', '--key', type=str)
    parser.add_argument('-f', '--force', action='store_true')

    args = parser.parse_args()

    process(**vars(args))
