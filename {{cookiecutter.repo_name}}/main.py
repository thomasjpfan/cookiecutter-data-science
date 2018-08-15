import argparse

from utils import get_params

from settings import EXPERIMENTS, PROCESS_FUNCS

PROCESS_CHOICES = [
    key.split("files__proc_", maxsplit=1)[1] for key in PROCESS_FUNCS
]


def run(args):
    experiment = args.experiment
    cmd = args.cmd
    run_id = args.run_id
    not_record_local = args.not_record_local

    exp = EXPERIMENTS[experiment]

    config = dict()
    config['record_local'] = not not_record_local
    if run_id:
        config['run_id'] = run_id

    exp.run(command_name=cmd, config_updates=config)


def process(args):
    key = args.key
    force = args.force

    params = get_params()

    key = f'files__proc_{key}'
    try:
        process_fn = params[key]
    except KeyError:
        print(f'files/processed/{key} does not exists in params')
        return

    if process_fn.exists() and not force:
        print(f'{process_fn} already exists, use --force')
        return

    print(f'Processing {key} filename: {process_fn}')
    PROCESS_FUNCS[key](params, force)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    process_parser = subparsers.add_parser('process')
    process_parser.add_argument(
        'key', choices=PROCESS_CHOICES, help='key of processed file')
    process_parser.add_argument(
        '-f', '--force', action='store_true', help='rerun processing for key')
    process_parser.set_defaults(func=process)

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument(
        'experiment', choices=EXPERIMENTS, help='key of model to run')
    run_parser.add_argument(
        'cmd',
        choices=['train', 'predict', 'train_hp'],
        help='command to run on model')
    run_parser.add_argument('-id', '--run-id', help='run id')
    run_parser.add_argument(
        '-nrl',
        '--not-record-local',
        action='store_true',
        help='do not record to local csv')
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)
