#!/usr/bin/env python3

"""Constant Model"""
import os

from sacred import Experiment
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats

from exp_utils import add_common_config
from files import RawFiles

exp = Experiment("linear_model")
add_common_config(exp, record_local=True)
rf = RawFiles("data")
LINEAR_MODEL = "linear_model.pkl"
PREDICT_FN = "predictions.npy"


@exp.command
def predict(model_id, run_dir, _log):
    model_fn = os.path.join(run_dir, LINEAR_MODEL)
    predict_fn = os.path.join(run_dir, PREDICT_FN)

    # Prediction task
    X = np.random.rand(100).reshape(-1, 1)
    linear_model = joblib.load(model_fn)
    y_predict = linear_model.predict(X)
    np.save(predict_fn, y_predict)

    _log.info(f"Finished prediction, model_id: {model_id}")


@exp.command
def train_hp(model_id, run_dir, _log, _run):
    X = np.random.rand(300).reshape(-1, 1)
    y = 4 * X + np.random.randn(300, 1) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = {"alpha": scipy.stats.uniform(0, 2)}
    rs = RandomizedSearchCV(Ridge(), params, n_iter=20, scoring='neg_mean_squared_error')
    rs.fit(X_train, y_train)

    train_score = mean_squared_error(y_train, rs.predict(X_train))
    test_score = mean_squared_error(y_test, rs.predict(X_test))

    _log.warning(
        f"Finished hyperparameter search model_id: {model_id}, test_score: "
        f"{test_score:0.6}, train_score: {train_score:0.6}, "
        f"params: {rs.best_params_}"
    )

    return [test_score, train_score]


@exp.automain
def train(model_id, run_dir, _log, _run):

    X = np.random.rand(300).reshape(-1, 1)
    y = 4 * X + np.random.randn(300, 1) * 0.5

    linear_model = Ridge()
    valid_scores = cross_val_score(
        linear_model, X, y, scoring='neg_mean_squared_error', cv=5)
    valid_score = -np.mean(valid_scores)
    valid_score_std = np.std(valid_scores)

    linear_model.fit(X, y)
    train_score = mean_squared_error(y, linear_model.predict(X))

    model_fn = os.path.join(run_dir, LINEAR_MODEL)
    joblib.dump(linear_model, model_fn)
    _run.add_artifact(model_fn)

    _log.warning(
        f"Finished training, model_id: {model_id}, val_score: "
        f"{valid_score:0.6}+/-{valid_score_std:0.6}, "
        f"train_score: {train_score:0.6}")
    # valid score/error, train score/error
    return [valid_score, train_score]
