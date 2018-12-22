import os
from functools import wraps
import datetime
from pathlib import Path
import argparse
import logging
import pandas as pd
from pprint import pformat

import yaml
from munch import munchify


def from_dataframe_cache(key):
    """Returns a decorator that wraps a function with signature,
    (params: dict, force: bool, kwargs) that returns a pandas
    Dataframe.

    The ``params[key]`` should be a :class:`pathlib.Path` to
    save and load the cached dataframe from.

    Parameters
    ----------
    key: str
        key to query ``params`` to get path of cache

    """

    def cache_decorator(f):
        @wraps(f)
        def wrapper(params, force=False, **kwargs):
            fn = params[key]
            is_feather = fn.suffix in [".fthr"]
            is_parq = fn.suffix in [".parq"]

            if not is_feather and not is_parq:
                raise ValueError(f"Unsupported data type: {fn}")

            if fn.exists() and not force:
                if is_feather:
                    return pd.read_feather(fn)
                else:
                    return pd.read_parquet(fn)
            output = f(params, force, **kwargs)
            if is_feather:
                output.to_feather(fn)
            else:
                output.to_parquet(fn)
            return output

        return wrapper

    return cache_decorator


def get_log_file_handler(log_fn, level=logging.INFO):
    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    return file_handler


def get_stream_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(stream_handler)

    return logger


def get_config(
        root_dir=".",
        config_fn="config.yaml",
        raw_root="data/raw",
        process_root="data/proc",
        raw_key_root="files__raw_",
        process_key_root="files__proc_",
):
    config_fn = os.path.join(root_dir, config_fn)
    with open(config_fn, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if key.startswith(raw_key_root):
            config[key] = Path(os.path.join(root_dir, raw_root, value))
        elif key.startswith(process_key_root):
            config[key] = Path(os.path.join(root_dir, process_root, value))

    return munchify(config)


def normalize_config(config, run_dir):
    output = {}
    for key, value in config.items():
        if key.endswith("_fn"):
            output[key] = Path(os.path.join(run_dir, value))
        else:
            output[key] = value
    return munchify(output)


def run_cli(
        func_dict,
        model_name,
        tags=None,
        root_dir=".",
        config_fn="config.yaml",
        raw_root="data/raw",
        process_root="data/proc",
):

    if tags is None:
        tags = []

    parser = argparse.ArgumentParser()

    func_choices = list(func_dict.keys())

    parser.add_argument(
        "cmd", choices=func_choices, help="command to run on model")
    parser.add_argument("-id", "--run-id", help="run id")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")

    args = parser.parse_args()
    debug = args.debug
    run_id = args.run_id or datetime.datetime.utcnow().strftime(
        "%Y-%m-%dT%H-%M-%S")

    config = get_config(
        root_dir=root_dir,
        config_fn=config_fn,
        raw_root=raw_root,
        process_root=process_root,
    )

    log_level = logging.DEBUG if debug else logging.INFO
    log = get_stream_logger(model_name, level=log_level)

    model_id = f"{model_name}_{run_id}"
    run_dir = os.path.join("artifacts", model_id)

    os.makedirs(run_dir, exist_ok=True)

    log_fn = os.path.join(run_dir, f"log_{args.cmd}.txt")
    file_hander = get_log_file_handler(log_fn)
    log.addHandler(file_hander)

    p = normalize_config(config, run_dir)

    model_config = {k: v for k, v in p.items() if k.startswith(model_name)}
    log.info(pformat(model_config))

    comet_exp = None
    if not debug and "COMET_API_KEY" in os.environ:
        from comet_ml import Experiment
        comet_exp = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name=p.name,
            log_code=False,
            log_graph=False,
            parse_args=False,
            auto_param_logging=False,
            auto_metric_logging=False)
        comet_exp.log_parameters(model_config)

    output = func_dict[args.cmd](model_id, p, run_dir, log, comet_exp) or {}

    if not debug:
        from mlflow import log_metric, log_artifact, log_param
        for k, v in output.items():
            log_metric(k, v)
        if comet_exp is not None:
            comet_exp.log_metrics(output)
        log_artifact(log_fn)
        log_param("model_id", model_id)


def get_classification_skorch_callbacks(model_id,
                                        pgroups,
                                        run_dir,
                                        comet_exp=None,
                                        per_epoch=True):

    from skorch.callbacks import EpochScoring, Checkpoint

    from mltome.skorch.callbacks import (
        LRRecorder,
        TensorboardXLogger,
    )

    pgroup_names = [item[0] + "_lr" for item in pgroups]
    tensorboard_log_dir = os.path.join("artifacts/runs", model_id)

    batch_targets = ["train_loss"]
    epoch_targets = ["train_acc", "valid_acc"]
    if per_epoch:
        epoch_targets.extend(pgroup_names)
    else:
        batch_targets.extend(pgroup_names)

    callbacks = [
        EpochScoring(
            "accuracy", name="train_acc", lower_is_better=False,
            on_train=True),
        LRRecorder(group_names=pgroup_names),
        TensorboardXLogger(
            tensorboard_log_dir,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets,
            epoch_groups=["acc"],
        ),
        Checkpoint(dirname=run_dir)
    ]

    if comet_exp is not None:
        from mltome.cometml import CometSkorchCallback

        neptune_callback = CometSkorchCallback(
            comet_exp,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets,
        )
        callbacks.append(neptune_callback)

    return callbacks
