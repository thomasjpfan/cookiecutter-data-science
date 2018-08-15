import os
from pathlib import Path
import datetime

import yaml
from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from munch import munchify

from mltome.sacred.config import (add_monogodb, add_neptune_observers,
                                  add_pushover_handler)

from mltome.sacred.observers import CSVObserver, ArtifactObserver
from mltome.logging import get_stream_logger

SETTINGS.CAPTURE_MODE = 'no'


def get_params(root_dir=".",
               config_fn="neptune.yaml",
               raw_root="data/raw",
               process_root="data/proc"):
    config_fn = os.path.join(root_dir, config_fn)
    with open(config_fn, "r") as f:
        config = yaml.safe_load(f)

    params = config['parameters']
    for key, value in params.items():
        if key.startswith("files__raw_"):
            config['parameters'][key] = Path(
                os.path.join(root_dir, raw_root, value))
        elif key.startswith("files__proc_"):
            config['parameters'][key] = Path(
                os.path.join(root_dir, process_root, value))

    return munchify(params)


def add_common_config(exp, csv_fn, record_local=True):
    exp.add_config(
        run_id=datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
        record_local=record_local,
        name=exp.path)

    @exp.config
    def run_dir_config(name, run_id):
        model_id = f"{name}_{run_id}"  # noqa
        run_dir = os.path.join('artifacts', model_id)  # noqa

    exp.logger = get_stream_logger(exp.path)
    exp.observers.append(CSVObserver(csv_fn))
    exp.observers.append(ArtifactObserver(exp.logger))
    exp.captured_out_filter = apply_backspaces_and_linefeeds


def normalize_params(params, run_dir):
    output = {}
    for key, value in params.items():
        if key.endswith("_fn"):
            output[key] = Path(os.path.join(run_dir, value))
        else:
            output[key] = value
    return munchify(output)


def generate_experiment_params_from_env(name,
                                        tags=None,
                                        record_local=True,
                                        csv_fn='artifacts/result.csv',
                                        root_dir=".",
                                        config_fn="neptune.yaml",
                                        raw_root="data/raw",
                                        process_root="data/proc"):
    if tags is None:
        tags = []
    exp = Experiment(name)
    params = get_params(
        root_dir=root_dir,
        config_fn=config_fn,
        raw_root=raw_root,
        process_root=process_root)

    str_params = {k: str(v) for k, v in params.items()}

    exp.add_config(**str_params)

    exp.add_config(tags=tags)
    add_common_config(exp, csv_fn, record_local=record_local)

    mongodb_url = os.environ.get('MONGODB_URL')
    mongodb_name = os.environ.get('MONGODB_NAME')
    pushover_user_token = os.environ.get('NOTIFIERS_PUSHOVER_USER')
    pushover_token = os.environ.get('NOTIFIERS_PUSHOVER_TOKEN')
    use_neptune = os.environ.get('USE_NEPTUNE') == 'true'

    neptune_ctx = None
    if use_neptune:
        from deepsense import neptune
        neptune_ctx = neptune.Context()

    add_monogodb(exp.observers, mongodb_url, mongodb_name)
    add_pushover_handler(exp.logger, pushover_user_token, pushover_token)
    add_neptune_observers(exp.observers, 'model_id', 'tags', ctx=neptune_ctx)

    return exp, params, neptune_ctx


def get_classification_skorch_callbacks(model_id,
                                        checkpoint_fn,
                                        history_fn,
                                        pgroups,
                                        log_func=print,
                                        neptune_ctx=None,
                                        per_epoch=True):

    from skorch.callbacks import EpochScoring
    from skorch.callbacks import Checkpoint

    from mltome.skorch.callbacks import (LRRecorder, HistorySaver,
                                         TensorboardXLogger)

    pgroup_names = [item[0] + "_lr" for item in pgroups]
    tensorboard_log_dir = os.path.join('artifacts/runs', model_id)

    batch_targets = ['train_loss']
    epoch_targets = ['train_acc', 'valid_acc']
    if per_epoch:
        epoch_targets.extend(pgroup_names)
    else:
        batch_targets.extend(pgroup_names)

    callbacks = [
        EpochScoring(
            'accuracy', name='train_acc', lower_is_better=False,
            on_train=True),
        LRRecorder(group_names=pgroup_names),
        TensorboardXLogger(
            tensorboard_log_dir,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets,
            epoch_groups=['acc']),
        Checkpoint(target=checkpoint_fn, sink=log_func),
        HistorySaver(target=history_fn)
    ]

    if neptune_ctx is not None:
        from mltome.neptune import NeptuneSkorchCallback
        neptune_callback = NeptuneSkorchCallback(
            neptune_ctx,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets)
        callbacks.append(neptune_callback)

    return callbacks
