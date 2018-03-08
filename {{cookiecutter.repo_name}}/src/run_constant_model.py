#!/usr/bin/env python3

"""Constant Model"""
import os

from sacred import Experiment
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np

from exp_utils import (
    add_common_config, setup_run_dir_predict, setup_run_dir_train
)
from files import RawFiles

exp = Experiment("Constant_Model")
add_common_config(exp, record_local=True)
rf = RawFiles("data")
CONSTANT_MODEL_FN = "constant_model.pkl"
VAL_TRAIN_SCORE = "val_train_score.txt"
PREDICT_FN = "predictions.npy"


class ConstantModel(BaseEstimator, RegressorMixin):

    def fit(self, X, y=None):
        y = check_array(y, copy=True, ensure_2d=False)
        self.constant_ = np.mean(y, axis=0).reshape(1, -1)
        return self

    def predict(self, X):
        check_is_fitted(self, 'constant_')
        return np.repeat(self.constant_, X.shape[0], axis=0)

    def predict_proba(self, X):
        check_is_fitted(self, 'constant_')
        return np.repeat(self.constant_, X.shape[0], axis=0)


@exp.command
def predict(run_id, _config, _log):
    run_dir = setup_run_dir_predict(run_id, _config, _log)
    model_fn = os.path.join(run_dir, CONSTANT_MODEL_FN)
    predict_fn = os.path.join(run_dir, PREDICT_FN)

    _log.info(f"Start prediction, run_id: {run_id}")
    # Prediction task
    X = np.random.rand(200)
    constant_model = joblib.load(model_fn)
    y_predict = constant_model.predict(X)
    np.save(predict_fn, y_predict)

    _log.info(f"Finished prediction, run_id: {run_id}")


@exp.automain
def train(run_id, _config, _log, _run):

    run_dir = setup_run_dir_train(run_id, _config, _log)
    _log.info(f"Start training, run_id: {run_id}")
    X, y = np.random.rand(200), np.random.randint(0, 2, 200)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    constant_model = ConstantModel()
    constant_model.fit(X_train, y_train)
    train_score = log_loss(y_train, constant_model.predict(X_train))
    test_score = log_loss(y_test, constant_model.predict(X_test))

    model_fn = os.path.join(run_dir, CONSTANT_MODEL_FN)
    joblib.dump(constant_model, model_fn)
    _run.add_artifact(model_fn)

    val_train_score_fn = os.path.join(run_dir, VAL_TRAIN_SCORE)
    val_train_score = np.array([test_score, train_score])
    np.savetxt(val_train_score_fn, val_train_score)

    _log.info(
        f"Finished training, run_id: {run_id}, val_score: {test_score:0.6}, "
        f"train_score: {train_score:0.6}")
    # valid score/error, train score/error
    return [test_score, train_score]
