from deepsense import neptune
from sacred.observers.base import RunObserver


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
