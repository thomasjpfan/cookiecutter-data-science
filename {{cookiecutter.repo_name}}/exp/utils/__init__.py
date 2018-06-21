import os

from skorch.callbacks import EpochScoring
from skorch.callbacks import Checkpoint

from .config import get_params, add_common_config
from .cache import from_dataframe_cache
from .tensorboard import TensorboardXRecorder

__all__ = ['get_params', 'add_common_config', 'from_dataframe_cache',
           'get_neptune_skorch_callback']


def get_neptune_skorch_callback(batch_targets=None, epoch_targets=None):
    use_neptune = os.environ.get('USE_NEPTUNE')

    if use_neptune != 'true':
        return None

    from exp.utils.neptune import NeptuneSkorchCallback
    return NeptuneSkorchCallback(batch_targets=batch_targets,
                                 epoch_targets=epoch_targets)


def get_classification_skorch_callbacks(
        model_id, checkpoint_fn):
    callbacks = [
        EpochScoring(
            'accuracy', name='train_acc', lower_is_better=False,
            on_train=True),
        TensorboardXRecorder(
            model_id,
            batch_targets=['train_loss'],
            epoch_targets=['train_acc', 'valid_acc']),
        Checkpoint(target=checkpoint_fn)
    ]

    neptune_callback = get_neptune_skorch_callback(
        batch_targets=['train_loss'], epoch_targets=['train_acc', 'valid_acc'])
    if neptune_callback is not None:
        callbacks.append(neptune_callback)

    return callbacks
