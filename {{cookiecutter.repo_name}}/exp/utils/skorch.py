from skorch.callbacks.base import Callback


class MetricsRecorder(Callback):

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
