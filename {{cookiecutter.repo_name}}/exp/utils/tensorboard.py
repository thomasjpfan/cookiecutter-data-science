from itertools import product
from collections import defaultdict
from contextlib import suppress
import os

from .skorch import MetricsLogger
from tensorboardX import SummaryWriter


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
