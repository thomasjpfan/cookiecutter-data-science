"""Utility functions for folder mangament"""
from typing import Optional
from pathlib import Path
from datetime import datetime
import logging


def get_logger(name, log_fn):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(logging.DEBUG)

    stream_handler.setFormatter(logging.Formatter('=> %(message)s'))
    file_handler.setFormatter(logging.Formatter('=> %(message)s'))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_run_dir(run_id: Optional[str] = None) -> Path:
    artifacts_dir = Path("artifacts")

    if not run_id:
        run_id = datetime.now().isoformat()

    run_folder = artifacts_dir / run_id
    if not run_folder.exists():
        run_folder.mkdir()
    return run_folder


def get_model_dir(run_dir: Path, model_name: str) -> Path:
    model_dir = run_dir / model_name
    if not model_dir.exists():
        model_dir.mkdir()
    return model_dir


def write_score(run_dir: Path, cv: float):
    cv_path = run_dir / "{}.score".format(cv)
    cv_path.touch()


def get_data_dir(data_type: str="root") -> Path:
    if data_type == "root":
        return Path("data")
    elif data_type in ["external", "interim", "processed", "raw"]:
        return Path("data") / data_type
    else:
        raise RuntimeError("Invalid data_type")
