"""EXP 001"""
from functools import partial
import click
from exp_manager import exp_manager


RUN_ID = "exp_001"
COMMENTS = "Experiment one!"


def _train(work_dir, log, seed):
    return 0.5, 0.8


def _predict(results_dir, log):
    pass


@click.group()
def cli():
    pass


@click.command()
@click.option("--seed", type=int, default=42)
@click.option("-f", "--force", is_flag=True)
def train(seed, force):
    _train_now = partial(_train, seed=seed)
    exp_manager.run_experiment(RUN_ID, COMMENTS, _train_now, force=force)


@click.command()
@click.option("-f", "--force", is_flag=True)
def predict(force):
    exp_manager.predict_with_experiment(RUN_ID, _predict, force=False)


if __name__ == '__main__':
    cli()
