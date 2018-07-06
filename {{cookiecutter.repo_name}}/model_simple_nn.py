"""Linear Model"""
import os

import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier

from mltome import get_classification_skorch_callbacks
from mltome.sacred import generate_experiment_params_from_env

exp, params = generate_experiment_params_from_env(
    "simple_nn_model", tags=["simple_nn_model"])


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
    model_fn = os.path.join(run_dir, params.simple_nn__model_fn)
    predict_fn = os.path.join(run_dir, params.simple_nn__prediction_fn)

    X, _ = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    net.initialize()
    net.load_params(model_fn)

    y_predict = net.predict_proba(X)
    np.save(predict_fn, y_predict)

    _log.info(f"Finished prediction, model_id: {model_id}")


@exp.command
def train_hp(model_id, _log, _run):
    _log.info(f"No hyper parameter training")


@exp.command
def train(model_id, run_dir, _log, _run):
    checkpoint_fn = os.path.join(run_dir, params.simple_nn__model_fn)
    history_fn = os.path.join(run_dir, params.simple_nn__history_fn)

    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    pgroups = [
        ('dense0.*', {
            'lr': 0.02
        }),
    ]
    net.set_params(optimizer__param_groups=pgroups)

    callbacks = get_classification_skorch_callbacks(model_id, checkpoint_fn,
                                                    history_fn, pgroups)

    net.callbacks.extend(callbacks)
    net.fit(X, y)

    valid_score = net.history[-1, 'valid_acc']
    train_score = net.history[-1, 'train_acc']

    _log.warning(f"Finished training, model_id: {model_id}, val_score: "
                 f"{valid_score:0.6}, "
                 f"train_score: {train_score:0.6}")
    # valid score/error, train score/error
    return [valid_score, train_score]
