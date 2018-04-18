"""
Create processed files
"""
from functools import wraps
import os
import click
import pandas as pd
from exp_utils import get_config


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


@from_cache('train')
def get_process_train(config, force=False):
    tr = pd.read_csv(config['files']['raw']['train'])
    return tr


@from_cache('test')
def get_process_test(config, force=False):
    te = pd.read_csv(config['files']['raw']['test'])
    return te


process_funcs = {
    "train": get_process_train,
    "test": get_process_test,
}


def process_key(config, key, force):
    try:
        process_fn = config['files']['processed'][key]
    except KeyError:
        click.echo(f'files/processed/{key} does not exists in config')
        return

    if os.path.exists(process_fn) and not force:
        click.echo(f'{process_fn} already exists, use --force')
        return

    click.echo(f'Processing {key} filename: {process_fn}')
    process_funcs[key](config, force)


@click.command()
@click.option('-k', '--key')
@click.option('-f', '--force', is_flag=True)
def process(key, force):
    config = get_config()

    if key:
        process_key(config, key, force)
        return

    for key in process_funcs:
        process_key(config, key, force)


if __name__ == '__main__':
    process()
