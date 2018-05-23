"""
Create processed files
"""
import click
import pandas as pd
from exp_utils import get_config, from_dataframe_cache


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


def process_key(config, key, force):
    if not key.startswith("proc_"):
        click.echo(f"{key} is not process key in config.yaml")
        return

    try:
        process_fn = config['files'][key]
    except KeyError:
        click.echo(f'files/processed/{key} does not exists in config')
        return

    if process_fn.exists() and not force:
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
