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


def get_final_run_dir() -> Path:
    artifacts_dir = Path("artifacts")
    run_folder = artifacts_dir / "final"
    if not run_folder.exists():
        run_folder.mkdir()
    return run_folder


def get_model_dir(run_dir: Path, model_name: str) -> Path:
    model_dir = run_dir / model_name
    if not model_dir.exists():
        model_dir.mkdir()
    return model_dir


def get_predict_file(run_dir: Path) -> Path:
    output = run_dir / f"prediction.csv"
    return output


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

