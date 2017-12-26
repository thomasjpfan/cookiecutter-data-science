"""EXP 001"""
from functools import partial
import click
from pathlib import Path
from exp_manager import ExperimentManager


RUN_ID = "exp_001"
COMMENTS = "Experiment one!"


def _train(work_dir, data_dir, log, seed):
    return 0.5, 0.8


def _predict(results_dir, data_dir, work_dir, log):
    return 0.8


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.pass_context
@click.option("--seed", type=int, default=42)
@click.option("-f", "--force", is_flag=True)
def train(ctx, seed, force):
    exp_manager = ctx.obj["exp_manager"]
    _train_now = partial(_train, seed=seed)

    exp_manager.run_experiment(RUN_ID, COMMENTS, _train_now, force=force)


@cli.command()
@click.pass_context
@click.option("-f", "--force", is_flag=True)
def predict(ctx, force):
    exp_manager = ctx.obj["exp_manager"]
    exp_manager.predict_with_experiment(RUN_ID, _predict, force=False)


if __name__ == '__main__':
    exp_manager = ExperimentManager(Path("artifacts"), Path("data"))
    cli(obj={"exp_manager": exp_manager})
