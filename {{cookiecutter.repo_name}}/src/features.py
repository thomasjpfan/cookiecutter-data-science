"""
Features
"""
import click
from exp_utils import get_config


@click.command()
def features():
    config = get_config()
    click.echo(config)


if __name__ == '__main__':
    features()
