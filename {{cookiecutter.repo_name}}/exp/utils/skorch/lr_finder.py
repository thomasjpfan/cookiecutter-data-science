from contextlib import suppress

import numpy as np
import matplotlib.pyplot as plt
from skorch.callbacks.base import Callback

from .callbacks import LRRecorder


class LRFinder(Callback):
    def __init__(self, start_lr, end_lr, warm_start=10, scale_linear=False):

        self.start_lrs = start_lr
        self.end_lrs = end_lr
        self.scale_linear = scale_linear

        self.lr_multiplier = None
        self.warm_start = 10
        self.final_valid_batch_idx = None

    def on_train_begin(self, net, X, **kwargs):
        self.best_loss = 1e9
        self.total_samples = len(X)
        self.best_batch_idx = 0

        optimizer = net.optimizer_
        self.start_lrs = self._format_lrs("start_lr", optimizer, self.start_lrs)
        self.end_lrs = self._format_lrs("end_lr", optimizer, self.end_lrs)
        self.optimizer = optimizer

    def on_batch_begin(self, net, X, **kwargs):
        if self.lr_multiplier is None:
            ratio = self.end_lrs/self.start_lrs
            num_of_batches = self.total_samples // len(X)

            self.lr_multiplier = ratio/num_of_batches
            if not self.scale_linear:
                self.lr_multiplier = ratio**(1/num_of_batches)

        batch_idx = self._get_batch_idx(net)
        self.batch_step(batch_idx)

    def on_batch_end(self, net, **kwargs):
        batch_idx = self._get_batch_idx(net)
        loss = net.history[-1, 'batches', 'train_loss', -1]
        if not np.isfinite(loss) or loss > 4*self.best_loss:
            raise ValueError("loss is too big")
        if loss < self.best_loss and batch_idx > self.warm_start:
            self.best_loss = loss
            self.best_lr = self.get_lr(batch_idx)

    def _format_lrs(self, name, optimizer, lr):
        if isinstance(lr, (list, tuple)):
            if len(lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} values for "
                    f"{name}, got {len(lr)}")
            return np.array(lr)
        else:
            return lr * np.ones(len(optimizer.param_groups))

    def batch_step(self, batch_idx):
        lrs = self.get_lr(batch_idx)
        pgroups_lr = zip(self.optimizer.param_groups, lrs)
        for param_group, lr in pgroups_lr:
            param_group['lr'] = lr

    def get_lr(self, batch_idx):
        mult = self.lr_multiplier * (batch_idx + 1)
        if not self.scale_linear:
            mult = self.lr_multiplier ** (batch_idx + 1)
        return self.start_lrs * mult

    def _get_batch_idx(self, net):
        if not net.history:
            return -1
        epoch = len(net.history) - 1
        current_batch_idx = len(net.history[-1, 'batches']) - 1
        batch_cnt = len(net.history[-2, 'batches']) if epoch >= 1 else 0
        return epoch * batch_cnt + current_batch_idx


def lr_find(net_cls, module, criterion, batch_size, X, y,
            start_lr=1e-5, end_lr=10, scale_linear=False):
    lr_finder = ('lr_finder',
                 LRFinder(start_lr=start_lr,
                          end_lr=end_lr,
                          scale_linear=scale_linear))
    lr_recorder = ('lr_recorder', LRRecorder())
    callbacks = [
        lr_finder, lr_recorder
    ]

    net = net_cls(module, criterion=criterion,
                  max_epochs=1,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  train_split=None)

    with suppress(ValueError):
        net.fit(X, y)

    return net, lr_finder[1]


def plot_lr(history, lr_finder, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    if not lr_finder.scale_linear:
        ax.set_xscale('log')

    n_skip = lr_finder.warm_start

    losses = history[-1, 'batches', 'train_loss'][n_skip:-1]
    lrs = history[-1, 'batches', 'default_lr'][n_skip:]
    ax.vlines(lr_finder.best_lr, min(losses), max(losses), color='r')
    ax.set_ylim(min(losses), max(losses))
    ax.set_xlabel("Learning rate (log-scaled)")
    ax.set_ylabel("Loss")
    ax.plot(lrs, losses)
