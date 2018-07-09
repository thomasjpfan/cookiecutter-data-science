import os
from sacred import Experiment

from mltome.sacred.config import (add_common_config, add_monogodb,
                                  add_neptune_observers, add_pushover_handler)
from mltome import get_params


def generate_experiment_params_from_env(name,
                                        tags=None,
                                        record_local=True,
                                        csv_fn='artifacts/result.csv',
                                        root_dir=".",
                                        config_fn="neptune.yaml",
                                        raw_root="data/raw",
                                        process_root="data/proc"):
    if tags is None:
        tags = []
    exp = Experiment(name)
    params = get_params(
        root_dir=root_dir,
        config_fn=config_fn,
        raw_root=raw_root,
        process_root=process_root)
    exp.add_config(**params)

    exp.add_config(tags=tags)
    add_common_config(exp, csv_fn, record_local=record_local)

    mongodb_url = os.environ.get('MONGODB_URL')
    mongodb_name = os.environ.get('MONGODB_NAME')
    pushover_user_token = os.environ.get('NOTIFIERS_PUSHOVER_USER')
    pushover_token = os.environ.get('NOTIFIERS_PUSHOVER_TOKEN')
    use_neptune = os.environ.get('USE_NEPTUNE') == 'true'

    neptune_ctx = None
    if use_neptune:
        from deepsense import neptune
        neptune_ctx = neptune.Context()

    add_monogodb(exp.observers, mongodb_url, mongodb_name)
    add_pushover_handler(exp.logger, pushover_user_token, pushover_token)
    add_neptune_observers(exp.observers, ctx=neptune_ctx)

    return exp, params, neptune_ctx
