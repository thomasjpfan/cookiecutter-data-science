from deepsense import neptune
from sacred.observers.base import RunObserver
from .skorch import MetricsRecorder


class NeptuneObserver(RunObserver):

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):
        self.ctx = neptune.Context()
        self.ctx.properties['model_id'] = config['model_id'] + '_' + command

        tags = config.get("tags") or []
        for tag in tags:
            self.ctx.tags.append(tag)

    def completed_event(self, stop_time, result):
        if not result or len(result) != 2:
            return
        self.ctx.channel_send("valid", 0, result[0])
        self.ctx.channel_send("train", 0, result[1])


class NeptuneSkorchCallback(MetricsRecorder):

    def __init__(self, batch_targets=None, epoch_targets=None):
        self.ctx = neptune.Context()

        super().__init__(
            batch_targets=batch_targets, epoch_targets=epoch_targets)

    def update_batch_values(self, values, idx):
        for name, value in values.items():
            self.ctx.channel_send(name, idx, value)

    def update_epoch_values(self, values, idx):
        for name, value in values.items():
            self.ctx.channel_send(name, idx, value)
