import os

from .skorch import MetricsRecorder
from tensorboardX import SummaryWriter


class TensorboardXRecorder(MetricsRecorder):

    def __init__(self, name,
                 batch_targets=None, epoch_targets=None,
                 root_dir="artifacts/runs"):
        log_dir = os.path.join(root_dir, name)
        self.writer = SummaryWriter(log_dir=log_dir)
        super().__init__(
            batch_targets=batch_targets, epoch_targets=epoch_targets)

    def update_batch_values(self, values, idx):
        for name, value in values.items():
            self.writer.add_scalar(f'batch/{name}', value, idx)
        if len(values) >= 2:
            self.writer.add_scalars('batch', values, idx)

    def update_epoch_values(self, values, idx):
        for name, value in values.items():
            self.writer.add_scalar(f'epoch/{name}', value, idx)
        if len(values) >= 2:
            self.writer.add_scalars('epoch', values, idx)

    def on_train_end(self, net, **kwargs):
        self.writer.close()
