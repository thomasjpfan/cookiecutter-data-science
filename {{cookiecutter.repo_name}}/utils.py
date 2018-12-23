from functools import wraps
import pandas as pd


def from_dataframe_cache(key):
    """Returns a decorator that wraps a function with signature,
    (params: dict, force: bool, kwargs) that returns a pandas
    Dataframe.

    The ``params[key]`` should be a :class:`pathlib.Path` to
    save and load the cached dataframe from.

    Parameters
    ----------
    key: str
        key to query ``params`` to get path of cache

    """

    def cache_decorator(f):
        @wraps(f)
        def wrapper(params, force=False, **kwargs):
            fn = params[key]
            is_feather = fn.suffix in [".fthr"]
            is_parq = fn.suffix in [".parq"]

            if not is_feather and not is_parq:
                raise ValueError(f"Unsupported data type: {fn}")

            if fn.exists() and not force:
                if is_feather:
                    return pd.read_feather(fn)
                else:
                    return pd.read_parquet(fn)
            output = f(params, force, **kwargs)
            if is_feather:
                output.to_feather(fn)
            else:
                output.to_parquet(fn)
            return output

        return wrapper

    return cache_decorator


def get_classification_skorch_callbacks(model_id,
                                        pgroups,
                                        run_dir,
                                        comet_exp=None,
                                        per_epoch=True):

    from skorch.callbacks import EpochScoring, Checkpoint

    from mltome.skorch.callbacks import (
        LRRecorder,
        # TensorboardXLogger,
    )

    pgroup_names = [item[0] + "_lr" for item in pgroups]
    # tensorboard_log_dir = os.path.join("artifacts/runs", model_id)

    batch_targets = ["train_loss"]
    epoch_targets = ["train_acc", "valid_acc"]
    if per_epoch:
        epoch_targets.extend(pgroup_names)
    else:
        batch_targets.extend(pgroup_names)

    callbacks = [
        EpochScoring(
            "accuracy", name="train_acc", lower_is_better=False,
            on_train=True),
        LRRecorder(group_names=pgroup_names),
        # TensorboardXLogger(
        #     tensorboard_log_dir,
        #     batch_targets=batch_targets,
        #     epoch_targets=epoch_targets,
        #     epoch_groups=["acc"],
        # ),
        Checkpoint(dirname=run_dir)
    ]

    if comet_exp is not None:
        from mltome.cometml import CometSkorchCallback

        neptune_callback = CometSkorchCallback(
            comet_exp,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets,
        )
        callbacks.append(neptune_callback)

    return callbacks
