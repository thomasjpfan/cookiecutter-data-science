import os
import datetime
from pathlib import Path
import argparse
import logging
from pprint import pformat

import yaml
from munch import munchify


def get_log_file_handler(log_fn, level=logging.INFO):
    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    return file_handler


def get_stream_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(stream_handler)

    return logger


def get_params(root_dir=".",
               config_fn="neptune.yaml",
               raw_root="data/raw",
               process_root="data/proc",
               raw_key_root="files__raw_",
               process_key_root="files__proc_"):
    config_fn = os.path.join(root_dir, config_fn)
    with open(config_fn, "r") as f:
        config = yaml.safe_load(f)

    params = config['parameters']
    for key, value in params.items():
        if key.startswith(raw_key_root):
            config['parameters'][key] = Path(
                os.path.join(root_dir, raw_root, value))
        elif key.startswith(process_key_root):
            config['parameters'][key] = Path(
                os.path.join(root_dir, process_root, value))

    return munchify(params)


def normalize_params(params, run_dir):
    output = {}
    for key, value in params.items():
        if key.endswith("_fn"):
            output[key] = Path(os.path.join(run_dir, value))
        else:
            output[key] = value
    return munchify(output)


def run_cli(func_dict,
            model_name,
            tags=None,
            root_dir=".",
            config_fn="neptune.yaml",
            raw_root="data/raw",
            process_root="data/proc"):

    if tags is None:
        tags = []

    parser = argparse.ArgumentParser()

    func_choices = list(func_dict.keys())

    parser.add_argument(
        'cmd', choices=func_choices, help='command to run on model')
    parser.add_argument('-id', '--run-id', help='run id')
    parser.add_argument('-d', '--debug', action='store_true', help='debug')

    args = parser.parse_args()
    debug = args.debug
    run_id = args.run_id or datetime.datetime.utcnow().strftime(
        '%Y-%m-%dT%H-%M-%S')

    params = get_params(
        root_dir=root_dir,
        config_fn=config_fn,
        raw_root=raw_root,
        process_root=process_root)

    log_level = logging.DEBUG if debug else logging.INFO
    log = get_stream_logger(model_name, level=log_level)

    model_id = f"{model_name}_{run_id}"
    run_dir = os.path.join('artifacts', model_id)

    os.makedirs(run_dir, exist_ok=True)

    log_fn = os.path.join(run_dir, f'log_{args.cmd}.txt')
    file_hander = get_log_file_handler(log_fn)
    log.addHandler(file_hander)

    p = normalize_params(params, run_dir)

    model_params = {k: v for k, v in p.items() if k.startswith(model_name)}
    log.info(pformat(model_params))

    neptune_ctx = None
    if "NEPTUNE_ONLINE_CONTEXT" in os.environ:
        import neptune
        neptune_ctx = neptune.Context()
        neptune_ctx.properties["model_id"] = model_id
        for tag in tags:
            neptune_ctx.tags.append(tag)

    output = func_dict[args.cmd](model_id, p, run_dir, log) or {}

    if not debug:
        from mlflow import log_metric, log_artifact, log_param
        for k, v in output.items():
            log_metric(k, v)
            if neptune_ctx is not None:
                neptune_ctx.channel_send(k, v)
        log_artifact(log_fn)
        log_param("model_id", model_id)


def get_classification_skorch_callbacks(model_id,
                                        checkpoint_fn,
                                        history_fn,
                                        pgroups,
                                        log_func=print,
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

    if "NEPTUNE_ONLINE_CONTEXT" in os.environ:
        from mltome.neptune import NeptuneSkorchCallback
        import neptune
        neptune_ctx = neptune.Context()
        neptune_callback = NeptuneSkorchCallback(
            neptune_ctx,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets)
        callbacks.append(neptune_callback)

    return callbacks
