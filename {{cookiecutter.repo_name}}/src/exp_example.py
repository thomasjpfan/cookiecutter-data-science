"""EXP EXAMPLE"""
from sacred import Experiment

from common import (
    add_common_config, setup_run_dir_predict, setup_run_dir_train
)

exp = Experiment("EXP_EXAMPLE")
add_common_config(exp, record_local=True)


@exp.command
def predict(run_id, _config, _log):
    setup_run_dir_predict(run_id, _config, _log)

    _log.info("Starting prediction")
    # Prediction task
    _log.info("Finished prediction")


@exp.automain
def train(run_id, _config, _log, _run):

    setup_run_dir_train(run_id, _config, _log)

    _log.info("Starting training")
    # Train task
    _log.info("Finished training")
    # valid error, train error
    return [0.5, 0.8]
