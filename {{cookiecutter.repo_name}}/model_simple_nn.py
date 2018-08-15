"""Linear Model"""
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier

from utils import (generate_experiment_params_from_env,
                   get_classification_skorch_callbacks, normalize_params)

exp, params, n_ctx = generate_experiment_params_from_env(
    "simple_nn", tags=["simple_nn"])


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


net = NeuralNetClassifier(MyModule, max_epochs=10, lr=0.1, callbacks=[])


@exp.command
def predict(model_id, run_dir, _log):
    p = normalize_params(params, run_dir)

    X, _ = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    net.initialize()
    net.load_params(p.simple_nn__model_fn)

    y_predict = net.predict_proba(X)
    np.save(p.simple_nn__prediction_fn, y_predict)

    _log.info(f"Finished prediction, model_id: {model_id}")


@exp.command
def train_hp(model_id, _log, _run):
    _log.info(f"No hyper parameter training")


@exp.command
def train(model_id, run_dir, _log, _run):
    p = normalize_params(params, run_dir)

    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    pgroups = [
        ('dense0.*', {
            'lr': 0.02
        }),
    ]

    net.set_params(optimizer__param_groups=pgroups)
    net.set_params(callbacks__print_log__sink=_log.info)

    callbacks = get_classification_skorch_callbacks(
        model_id,
        p.simple_nn__model_fn,
        p.simple_nn__history_fn,
        pgroups,
        log_func=_log.info,
        neptune_ctx=n_ctx)

    net.callbacks.extend(callbacks)
    net.fit(X, y)

    valid_score = net.history[-1, 'valid_acc']
    train_score = net.history[-1, 'train_acc']

    _log.info(f"Finished training, model_id: {model_id}, val_score: "
              f"{valid_score:0.6}, "
              f"train_score: {train_score:0.6}")
    # valid score/error, train score/error
    return [valid_score, train_score]
