"""Constant Model"""
import os

from sacred import Experiment
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np

from common import (
    add_common_config, setup_run_dir_predict, setup_run_dir_train
)
from files import RawFiles

exp = Experiment("Constant_Model")
add_common_config(exp, record_local=True)
rf = RawFiles("data")
CONSTANT_MODEL_FN = "constant_model.pkl"
PREDICT_FN = "predictions.npy"


class ConstantModel(BaseEstimator):

    def fit(self, X, y=None):
        self.constant_ = np.mean(y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'constant_')
        return np.repeat(self.constant_, X.shape[0])


@exp.command
def predict(run_id, _config, _log):
    run_dir = setup_run_dir_predict(run_id, _config, _log)
    model_fn = os.path.join(run_dir, CONSTANT_MODEL_FN)
    predict_fn = os.path.join(run_dir, PREDICT_FN)

    _log.info("Starting prediction, run_dir: %s", run_dir)
    # Prediction task
    X = np.random.rand(200)
    constant_model = joblib.load(model_fn)
    y_predict = constant_model.predict(X)
    np.save(predict_fn, y_predict)

    _log.info("Finished prediction, run_dir: %s", run_dir)


@exp.automain
def train(run_id, _config, _log, _run):

    run_dir = setup_run_dir_train(run_id, _config, _log)
    _log.info("Starting training, run_dir: %s", run_dir)
    X, y = np.random.rand(200), np.random.randint(0, 2, 200)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    constant_model = ConstantModel()
    constant_model.fit(X_train, y_train)
    train_score = log_loss(y_train, constant_model.predict(X_train))
    test_score = log_loss(y_test, constant_model.predict(X_test))

    model_fn = os.path.join(run_dir, CONSTANT_MODEL_FN)
    joblib.dump(constant_model, model_fn)

    _log.info("Finished training, run_dir: % s", run_dir)
    # valid score/error, train score/error
    return [test_score, train_score]
