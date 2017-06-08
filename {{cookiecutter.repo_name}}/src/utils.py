"""Utility functions for folder mangament"""
from pathlib import Path
from datetime import datetime


def get_run_dir() -> Path:
    artifacts_dir = Path("artifacts")
    date_id = datetime.now().isoformat()
    run_folder = artifacts_dir / date_id
    if not run_folder.exists():
        run_folder.mkdir()
    return run_folder


def get_model_dir(run_dir: Path, model_name: str) -> Path:
    model_dir = run_dir / model_name
    if not model_dir.exists():
        model_dir.mkdir()
    return model_dir


def get_predict_file(run_dir: Path) -> Path:
    output = run_dir / f"submission.csv"
    return output


def write_cv(run_dir: Path, cv: float):
    cv_path = run_dir / "{}.txt".format(cv)
    cv_path.touch()


def get_data_dir() -> Path:
    return Path("data")


def get_processed_dir() -> Path:
    return Path("data") / "processed"


def get_final_run_dir() -> Path:
    artifacts_dir = Path("artifacts")
    run_folder = artifacts_dir / "final"
    if not run_folder.exists():
        run_folder.mkdir()
    return run_folder
