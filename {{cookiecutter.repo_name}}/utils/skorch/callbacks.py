from itertools import product
from collections import defaultdict
from contextlib import suppress
import os

from skorch.callbacks.base import Callback
from tensorboardX import SummaryWriter


class MetricsLogger(Callback):

    def __init__(self, batch_targets=None, epoch_targets=None):

        self.batch_targets = batch_targets
        self.epoch_targets = epoch_targets

    def update_batch_values(self, values, idx):
        raise NotImplementedError("Not implemented")

    def update_epoch_values(self, values, idx):
        raise NotImplementedError("Not implemented")

    def on_batch_end(self, net, **kwargs):
        if self.batch_targets is None:
            return
        batch_idx = self._get_batch_idx(net)

        values = {}
        for name in self.batch_targets:
            values[name] = net.history[-1, 'batches', name, -1]

        self.update_batch_values(values, batch_idx)

    def on_epoch_end(self, net, **kwargs):
        if self.epoch_targets is None:
            return

        epoch = len(net.history)

        values = {}
        for name in self.epoch_targets:
            values[name] = net.history[-1, name]

        self.update_epoch_values(values, epoch)

    def _get_batch_idx(self, net):
        if not net.history:
            return -1
        epoch = len(net.history) - 1
        current_batch_idx = len(net.history[-1, 'batches']) - 1
        batch_cnt = len(net.history[-2, 'batches']) if epoch >= 1 else 0
        return epoch * batch_cnt + current_batch_idx


class LRRecorder(Callback):

    def __init__(self, group_names=None, per_epoch=True, default_group="default_lr"):
        if group_names is None:
            group_names = []
        if default_group:
            group_names.append(default_group)
        self.group_names = group_names
        self.per_epoch = per_epoch

    def on_train_begin(self, net, **kwargs):
        self.optimizer_ = net.optimizer_

    def on_epoch_end(self, net, **kwargs):
        if not self.per_epoch:
            return
        history = net.history
        pgroups = self.optimizer_.param_groups

        for pgroup, name in zip(pgroups, self.group_names):
            history.record(name, pgroup['lr'])

    def on_batch_end(self, net, **kwargs):
        if self.per_epoch:
            return
        history = net.history
        pgroups = self.optimizer_.param_groups

        for pgroup, name in zip(pgroups, self.group_names):
            history.record_batch(name, pgroup['lr'])


class HistorySaver(Callback):
    def __init__(self, target):
        self.target = target

    def on_epoch_end(self, net, **kwargs):
        net.save_history(self.target)


class TensorboardXLogger(MetricsLogger):

    def __init__(self, name,
                 batch_targets=None, epoch_targets=None,
                 batch_groups=None, epoch_groups=None,
                 root_dir='artifacts/runs'):
        log_dir = os.path.join(root_dir, name)
        self.writer = SummaryWriter(log_dir=log_dir)

        batch_groups = batch_groups or []
        epoch_groups = epoch_groups or []
        batch_targets = batch_targets or []
        epoch_targets = epoch_targets or []

        self.batch_target_to_name = {}
        for g, t in product(batch_groups, batch_targets):
            if t.endswith(g):
                self.batch_target_to_name[t] = 'batch_' + g

        self.epoch_target_to_name = {}
        for g, t in product(epoch_groups, epoch_targets):
            if t.endswith(g):
                self.epoch_target_to_name[t] = 'epoch_' + g

        super().__init__(
            batch_targets=batch_targets, epoch_targets=epoch_targets)

    def update_batch_values(self, values, idx):
        vgroups = defaultdict(dict)
        for name, value in values.items():
            self.writer.add_scalar(f'batch/{name}', value, idx)
            with suppress(KeyError):
                group = self.batch_target_to_name[name]
                vgroups[group][name] = value

        for group, values in vgroups.items():
            self.writer.add_scalars(group, values, idx)

    def update_epoch_values(self, values, idx):
        vgroups = defaultdict(dict)
        for name, value in values.items():
            self.writer.add_scalar(f'epoch/{name}', value, idx)
            with suppress(KeyError):
                group = self.epoch_target_to_name[name]
                vgroups[group][name] = value

        for group, values in vgroups.items():
            self.writer.add_scalars(group, values, idx)

    def on_train_end(self, net, **kwargs):
        self.writer.close()
