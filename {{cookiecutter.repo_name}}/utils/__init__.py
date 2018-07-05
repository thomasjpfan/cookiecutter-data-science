import os

from skorch.callbacks import EpochScoring
from skorch.callbacks import Checkpoint

from .skorch import LRRecorder, HistorySaver, TensorboardXLogger


def get_neptune_skorch_callback(batch_targets=None, epoch_targets=None):
    use_neptune = os.environ.get('USE_NEPTUNE')

    if use_neptune != 'true':
        return None

    from exp.utils.neptune import NeptuneSkorchCallback
    return NeptuneSkorchCallback(batch_targets=batch_targets,
                                 epoch_targets=epoch_targets)


def get_classification_skorch_callbacks(
        model_id, checkpoint_fn, history_fn, pgroups, per_epoch=True):

    pgroup_names = [item[0] + "_lr" for item in pgroups]

    batch_targets = ['train_loss']
    epoch_targets = ['train_acc', 'valid_acc']
    if per_epoch:
        epoch_targets.extend(pgroup_names)
    else:
        batch_targets.extend(pgroup_names)

    callbacks = [
        EpochScoring(
            'accuracy', name='train_acc', lower_is_better=False,
            on_train=True),
        LRRecorder(group_names=pgroup_names),
        TensorboardXLogger(
            model_id,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets,
            epoch_groups=['acc']),
        Checkpoint(target=checkpoint_fn),
        HistorySaver(target=history_fn)
    ]

    neptune_callback = get_neptune_skorch_callback(
        batch_targets=batch_targets, epoch_targets=epoch_targets)
    if neptune_callback is not None:
        callbacks.append(neptune_callback)

    return callbacks
