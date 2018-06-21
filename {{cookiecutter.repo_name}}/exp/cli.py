import click

from .linear_model import exp as linear_exp
from .simple_nn_model import exp as simple_nn_exp
from .utils import get_params
from .process import get_train, get_test

EXPERIMENTS = {"linear_model": linear_exp, "simple_nn_model": simple_nn_exp}
PROCESS_FUNCS = {
    "files__proc_train": get_train,
    "files__proc_test": get_test,
}
PROCESS_CHOICES = [
    key.split("files__proc_", maxsplit=1)[1] for key in PROCESS_FUNCS
]


@click.argument('experiment', type=click.Choice(EXPERIMENTS))
@click.argument('cmd', type=click.Choice(['train', 'predict', 'train_hp']))
@click.option('-id', '--run-id', help='run id')
@click.option(
    '-rl', '--record-local', help='record to local csv', is_flag=True)
def run(experiment, cmd, run_id, record_local):
    exp = EXPERIMENTS[experiment]

    config = dict()
    config['record_local'] = record_local
    if run_id:
        config['run_id'] = run_id

    exp.run(command_name=cmd, config_updates=config)


@click.argument('key', type=click.Choice(PROCESS_CHOICES))
@click.option('-f', '--force', is_flag=True)
def process(key, force):
    params = get_params()

    key = f'files__proc_{key}'
    try:
        process_fn = params[key]
    except KeyError:
        click.echo(f'files/processed/{key} does not exists in params')
        return

    if process_fn.exists() and not force:
        click.echo(f'{process_fn} already exists, use --force')
        return

    click.echo(f'Processing {key} filename: {process_fn}')
    PROCESS_FUNCS[key](params, force)
