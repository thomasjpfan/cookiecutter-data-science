"""First experimental model"""
from pathlib import Path

import click

from utils import (
    get_run_dir, get_model_dir, write_score,
    get_predict_file,
    get_data_dir
)

MODEL_NAME = "model0"


def _load_model():
    pass


def _train(model_dir: Path, run_dir: Path, run_id: str, seed: int) -> float:
    pass


def _predict(model_dir: Path, run_id: str, predict_file: Path):
    pass


@click.group()
def cli():
    pass


@cli.command()
@click.option("--run-id", type=str, default="")
@click.option("--seed", type=int, default=42)
def train(run_id, seed):

    run_dir = get_run_dir(run_id)

    model_dir = get_model_dir(run_dir, MODEL_NAME)

    print("Starting to train model")
    cv = _train(model_dir, run_dir, run_id, seed)

    print(f"Finish training {MODEL_NAME}, cv: {cv}")
    write_score(run_dir, cv)


@cli.command()
@click.argument("--run-dir", type=click.Path(exists=True))
def predict(run_dir):
    model_dir = get_model_dir(run_dir, MODEL_NAME)

    print("Predicting on test data!")

    predict_file = get_predict_file(run_dir)
    _predict(model_dir, predict_file)
    print("Done!")


if __name__ == '__main__':
    cli()
