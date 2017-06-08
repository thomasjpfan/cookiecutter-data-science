"""Features that are indepdent"""
import click
import pandas as pd

from utils import get_data_dir


def _gen_features(df: pd.DataFrame) -> pd.DataFrame:
    pass


@click.command()
def gen_features():
    data_dir = get_data_dir()
    pass
