"""
Create processed files
"""
import os
import click
from exp_utils import get_config

config = get_config()
processed_keys = list(config['files']['processed'].keys())


def process_train():
    pass


process_funcs = {
    "train": process_train,
}


def process_key(key, force):
    try:
        process_fn = config['files']['processed'][key]
    except KeyError:
        click.echo(f'processed/{key} does not exists in config')
        return

    if os.path.exists(process_fn) and not force:
        click.echo(f'{process_fn} already exists, use --force')
        return

    click.echo(f'Processing {key}')
    process_funcs[key]()
    click.echo(f'{process_fn} created')


@click.command()
@click.option('-k', '--key',
              type=click.Choice(processed_keys))
@click.option('-f', '--force', is_flag=True)
def process(key, force):

    if key:
        process_key(key, force)
        return

    # process everything
    for key_type in processed_keys:
        process_key(key_type, force)


if __name__ == '__main__':
    process()
