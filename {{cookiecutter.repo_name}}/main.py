import click

from mltome import get_params

from model_linear import exp as linear_exp
from model_simple_nn import exp as simple_nn_exp
from model_text import exp as text_exp
from process import get_train, get_test

EXPERIMENTS = {
    "linear": linear_exp,
    "simple_nn": simple_nn_exp,
    "text": text_exp,
}
PROCESS_FUNCS = {
    "files__proc_train": get_train,
    "files__proc_test": get_test,
}
PROCESS_CHOICES = [
    key.split("files__proc_", maxsplit=1)[1] for key in PROCESS_FUNCS
]


@click.group()
def cli():
    pass


@cli.command()
@click.argument('experiment', type=click.Choice(EXPERIMENTS))
@click.argument('cmd', type=click.Choice(['train', 'predict', 'train_hp']))
@click.option('-id', '--run-id', help='run id')
@click.option(
    '-nrl',
    '--not-record-local',
    help='do not record to local csv',
    is_flag=True)
def run(experiment, cmd, run_id, not_record_local):
    exp = EXPERIMENTS[experiment]

    config = dict()
    config['record_local'] = not not_record_local
    if run_id:
        config['run_id'] = run_id

    exp.run(command_name=cmd, config_updates=config)


@cli.command()
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


if __name__ == '__main__':
    cli()
