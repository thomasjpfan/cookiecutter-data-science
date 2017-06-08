"""First experimental model"""
from pathlib import Path

import click

from utils import (
    get_data_dir, get_run_dir, get_model_dir, write_cv,
    get_predict_file, get_final_run_dir
)

MODEL_NAME = "model0"


def _load_model():
    pass


def _train(data_dir: Path, model_dir: Path, run_dir: Path) -> float:
    pass


def _predict(data_dir: Path, model_dir: Path, predict_file: Path):
    pass


@click.group()
def cli():
    pass


@cli.command()
@click.option("final", is_flag=True)
def train(final):
    data_dir = get_data_dir()

    if final:
        run_dir = get_final_run_dir()
    else:
        run_dir = get_run_dir()

    model_dir = get_model_dir(run_dir, MODEL_NAME)

    print("Starting to train model")
    cv = _train(data_dir, model_dir, run_dir)

    print(f"Finish training {MODEL_NAME}, cv: {cv}")
    write_cv(run_dir, cv)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
def predict(run_dir):
    data_dir = get_data_dir()
    model_dir = get_model_dir(run_dir, MODEL_NAME)

    print("Predicting on test data!")

    predict_file = get_predict_file(run_dir)
    _predict(data_dir, model_dir, predict_file)
    print("Done!")
