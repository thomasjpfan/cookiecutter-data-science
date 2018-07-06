"""Linear Model"""
import os

from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from dask_ml.model_selection import RandomizedSearchCV
import numpy as np
import scipy.stats

from mltome.sacred import generate_experiment_params_from_env

exp, params = generate_experiment_params_from_env(
    "linear_model", tags=["linear_model"])

# from dask.distributed import Client
# client = Client('192.168.2.24:8786')


@exp.command
def predict(model_id, run_dir, _log):
    model_fn = os.path.join(run_dir, params.ridge__model_fn)
    predict_fn = os.path.join(run_dir, params.ridge__prediction_fn)

    # Prediction task
    X = np.random.rand(100).reshape(-1, 1)
    linear_model = joblib.load(model_fn)
    y_predict = linear_model.predict(X)
    np.save(predict_fn, y_predict)

    _log.info(f"Finished prediction, model_id: {model_id}")


@exp.command
def train_hp(model_id, _log, _run):
    X = np.random.rand(300).reshape(-1, 1)
    y = 4 * X + np.random.randn(300, 1) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = {"alpha": scipy.stats.uniform(0, 2)}
    rs = RandomizedSearchCV(
        Ridge(), params, n_iter=20, scoring='neg_mean_squared_error')
    rs.fit(X_train, y_train)

    train_score = mean_squared_error(y_train, rs.predict(X_train))
    test_score = mean_squared_error(y_test, rs.predict(X_test))

    _log.warning(
        f"Finished hyperparameter search model_id: {model_id}, test_score: "
        f"{test_score:0.6}, train_score: {train_score:0.6}, "
        f"params: {rs.best_params_}")

    return [test_score, train_score]


@exp.command
def train(model_id, run_dir, _log, _run):

    X = np.random.rand(300).reshape(-1, 1)
    y = 4 * X + np.random.randn(300, 1) * 0.5

    linear_model = Ridge(params.ridge__alpha)
    valid_scores = cross_val_score(
        linear_model, X, y, scoring='neg_mean_squared_error', cv=5)
    valid_score = -np.mean(valid_scores)
    valid_score_std = np.std(valid_scores)

    linear_model.fit(X, y)
    train_score = mean_squared_error(y, linear_model.predict(X))

    model_fn = os.path.join(run_dir, params.ridge__model_fn)
    joblib.dump(linear_model, model_fn)
    _run.add_artifact(model_fn)

    _log.warning(f"Finished training, model_id: {model_id}, val_score: "
                 f"{valid_score:0.6}+/-{valid_score_std:0.6}, "
                 f"train_score: {train_score:0.6}")
    # valid score/error, train score/error
    return [valid_score, train_score]
