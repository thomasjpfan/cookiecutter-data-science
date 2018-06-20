import logging
import os
import datetime

from pathlib import Path
from munch import munchify
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver
import yaml

from .observers import CSVObserver, ArtifactObserver
from .logging import get_stream_logger


def get_params(root_dir="."):
    config_fn = os.path.join(root_dir, "neptune.yaml")
    with open(config_fn, "r") as f:
        config = yaml.load(f)

    params = config['parameters']
    for key, value in params.items():
        if key.startswith("files__raw_"):
            config['parameters'][key] = Path(
                os.path.join(root_dir, "data/raw", value))
        elif key.startswith("files__proc_"):
            config['parameters'][key] = Path(
                os.path.join(root_dir, "data/proc", value))

    return munchify(params)


def add_common_config(exp, record_local=True):
    exp.add_config(
        run_id=datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
        record_local=record_local,
        name=exp.path
    )
    params = get_params()
    exp.add_config(**params)

    @exp.config
    def run_dir_config(name, run_id):
        model_id = f"{name}_{run_id}"  # noqa
        run_dir = os.path.join('artifacts', model_id)  # noqa

    exp.logger = get_stream_logger(exp.path)
    exp.observers.append(CSVObserver())
    exp.observers.append(ArtifactObserver(exp.logger))
    add_monogodb_from_env(exp.observers)
    add_pushover_handler_from_env(exp.logger)
    exp.captured_out_filter = apply_backspaces_and_linefeeds


def add_monogodb_from_env(exp):
    mongodb_url = os.environ.get('MONGODB_URL')
    mongodb_name = os.environ.get('MONGODB_NAME')

    if mongodb_url and mongodb_name:
        exp.append(MongoObserver.create(
            url=mongodb_url,
            db_name=mongodb_name
        ))


def add_pushover_handler_from_env(log):
    pushover_user_token = os.environ.get('NOTIFIERS_PUSHOVER_USER')
    pushover_token = os.environ.get('NOTIFIERS_PUSHOVER_TOKEN')

    if pushover_user_token and pushover_token:
        from notifiers.logging import NotificationHandler
        h = NotificationHandler('pushover')
        h.setLevel(logging.WARNING)
        log.addHandler(h)
