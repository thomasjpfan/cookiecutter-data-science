"""Linear Model"""
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint

from utils import get_classification_skorch_callbacks

from runner import get_runner
import fire


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
pgroups = [
    ('dense0.*', {
        'lr': 0.02
    }),
]


def predict(run):

    X, _ = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)

    cp = Checkpoint(dirname=run.run_dir)
    net.set_params(optimizer__param_groups=pgroups)
    net.initialize()
    net.load_params(checkpoint=cp)

    y_predict = net.predict_proba(X)
    np.save(run.cfg.simple_nn__prediction_fn, y_predict)

    run.log.info(f"Finished prediction: {run.model_id}")


def train(run):

    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    net.set_params(optimizer__param_groups=pgroups)
    net.set_params(callbacks__print_log__sink=run.log.info)

    callbacks = get_classification_skorch_callbacks(
        run.model_id, pgroups, run.run_dir, comet_exp=run.comet_exp)

    net.callbacks.extend(callbacks)
    net.fit(X, y)

    valid_score = net.history[-1, 'valid_acc']
    train_score = net.history[-1, 'train_acc']

    run.log.info(f"Finished training, model_id: {run.model_id}, val_score: "
                 f"{valid_score:0.6}, "
                 f"train_score: {train_score:0.6}")
    return {"valid": valid_score, "train": train_score}


if __name__ == '__main__':
    r = get_runner("simple_nn", [train, predict])
    fire.Fire(r)
