"""
Cleaning data with high bias, i.e. Uses domain
knowledge to clean data. This cleaning step will
not be be machine learned.
"""
import click
import pandas as pd

from utils import get_data_dir


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    pass


@click.command()
def clean():
    data_dir = get_data_dir()
    pass


if __name__ == '__main__':
    clean()
