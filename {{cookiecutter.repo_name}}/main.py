#!/usr/bin/env python3
import click

from exp.cli import run, process


@click.group()
def cli():
    pass


cli.command()(run)
cli.command()(process)

if __name__ == '__main__':
    cli()
