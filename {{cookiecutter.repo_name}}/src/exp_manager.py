"""Manages projects"""
from datetime import datetime
from pathlib import Path
import logging
import csv


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


def get_data_dir(self, data_type):
    if data_type == "root":
        return self.data_dir
    elif data_type in ["external", "interim", "processed", "raw"]:
        return self.data_dir / data_type
    else:
        raise RuntimeError("Invalid data_type for dir")


class ExperimentManager:

    COLS = ["run_id", "start_time", "end_time", "delta_time",
            "train_score", "valid_score", "comments"]

    def __init__(self, root_dir, data_dir):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.results_file = root_dir / "results.csv"
        if not self.results_file.exists():
            with self.results_file.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLS, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()

    def run_experiment(self, run_id, comments, train, force=False):
        work_dir = self.root_dir / run_id
        if not force and self.root_dir.exists():
            message = f"{work_dir} exists!"
            raise ValueError(message)

        log = get_logger(run_id, work_dir)
        log.info(f"Creating directory at {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=False)

        start_time = datetime.utcnow()
        log.info(f"Starting experiemnt {run_id} at {start_time}")

        train_score, valid_score = train(work_dir, log)

        end_time = datetime.utcnow()
        delta_time = (end_time - start_time).seconds / 60
        log.info(f"Finished experiment {run_id} at {end_time} delta_time={delta_time} min")
        log.info(f"Results: train_score={train_score} valid_score={valid_score}")

        self._save_results(self, run_id, train_score, valid_score,
                           start_time, end_time, delta_time, comments)
        log.info(f"Experiment results {run_id} saved")

    def predict_with_experiment(self, run_id, predict, force=False):
        work_dir = self.root_dir / run_id
        if not work_dir.exists():
            raise ValueError(f"{work_dir} does not exist!")

        results_dir = work_dir / "results"
        if not force and results_dir.exists():
            raise ValueError(f"{results_dir} exists!")
        log = get_logger(run_id, work_dir)

        start_time = datetime.utcnow()
        log.info(f"Starting prediction {run_id} at {start_time}")

        predict(results_dir, log)

        end_time = datetime.utcnow()
        delta_time = (end_time - start_time).seconds / 60
        log.info(f"Finished experiment {run_id} as {end_time} delta_time={delta_time} min")

    def _save_results(self, run_id, train_score, valid_score,
                      start_time, end_time, delta_time, comments):
        result = {"run_id": run_id, "start_time": f"{start_time}",
                  "end_time": f"{end_time}", "delta_time": delta_time,
                  "train_score": train_score, "valid_score": valid_score,
                  "comments": comments}
        with self.results_file.open("a") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLS, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(result)


exp_manager = ExperimentManager(Path("artifacts"), Path("data"))
