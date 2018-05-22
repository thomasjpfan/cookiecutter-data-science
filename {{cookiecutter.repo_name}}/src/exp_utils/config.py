from contextlib import suppress
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

TIME_FORMAT = '%Y%m%d_%H%M%S'


def get_config(root_dir="."):
    config_fn = os.path.join(root_dir, "config.yaml")
    with open(config_fn, "r") as f:
        config = yaml.load(f)

    if root_dir == ".":
        return config

    with suppress(KeyError):
        files = config['files']
        for file_key, file_path in files.items():
            config['files'][file_key] = Path(os.path.join(root_dir, file_path))

    return munchify(config)


def add_common_config(exp, record_local=True):
    exp.add_config(
        run_id=datetime.datetime.utcnow().strftime(TIME_FORMAT),
        record_local=record_local,
        name=exp.path
    )
    exp.add_config("config.yaml")

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
