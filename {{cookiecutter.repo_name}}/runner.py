from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from pprint import pformat
import os

import yaml
from munch import munchify


def get_config(
        root_dir=".",
        config_fn="config.yaml",
        raw_root="data/raw",
        process_root="data/proc",
        raw_key_root="files__raw_",
        process_key_root="files__proc_",
):
    config_fn = Path(root_dir) / config_fn
    root_dir = Path(root_dir)
    with config_fn.open("r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if key.startswith(raw_key_root):
            config[key] = root_dir / raw_root / value
        elif key.startswith(process_key_root):
            config[key] = root_dir / process_root / value

    return munchify(config)


def normalize_config(config, run_dir):
    output = {}
    for key, value in config.items():
        if key.endswith("_fn"):
            output[key] = run_dir / value
        else:
            output[key] = value
    return munchify(output)


class Runner(ABC):
    def __init__(self, rid=None, debug=True):
        run_id = rid or datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        model_id = f"{self.name}_{run_id}"

        self.model_id = model_id

        # Set up logger
        log = logging.getLogger(self.name)
        level = logging.DEBUG if debug else logging.INFO
        log.setLevel(level)

        formatter = logging.Formatter("%(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        log.addHandler(stream_handler)

        self.log = log
        self.log_level = level
        self.log_formatter = formatter

        # Create artifacts dir
        run_dir = Path("artifacts") / model_id

        self.run_dir = run_dir

        config = get_config()
        config = normalize_config(config, run_dir)

        model_config = {
            k: v
            for k, v in config.items() if k.startswith(self.name)
        }
        log.info(pformat(model_config))

        self.cfg = config

        comet_exp = None
        if not debug and "COMET_API_KEY" in os.environ:
            from comet_ml import Experiment
            comet_exp = Experiment(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name=config.name,
                log_code=False,
                log_graph=False,
                parse_args=False,
                auto_param_logging=False,
                auto_metric_logging=False)
            comet_exp.log_parameters(model_config)

        self.comet_exp = comet_exp
        self.debug = debug

    def add_file_logger(self, func_name):
        self.run_dir.mkdir(exist_ok=True)

        log_path = self.run_dir / f"log_{func_name}"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.log_formatter)
        self.log.addHandler(file_handler)
        self.log_path = log_path

    def log_results(self, results):
        if not self.debug:
            from mlflow import log_metric, log_artifact, log_param
            for k, v in results.items():
                log_metric(k, v)
            if self.comet_exp is not None:
                self.comet_exp.log_metrics(results)
            log_artifact(self.log_path)
            log_param("model_id", self.model_id)

    @property
    @abstractmethod
    def name():
        ...


def gen_inner_func(func, name):
    def inner_func(run):
        run.add_file_logger(name)
        results = func(run)
        if results is not None:
            run.log_results(results)

    return inner_func


def get_runner(model_nane, funcs=None):
    class NamedRunner(Runner):
        @property
        def name(self):
            return model_nane

    funcs = funcs or []

    for func in funcs:
        inner_func = gen_inner_func(func, func.__name__)
        setattr(NamedRunner, func.__name__, inner_func)

    Runner.register(NamedRunner)

    return NamedRunner
