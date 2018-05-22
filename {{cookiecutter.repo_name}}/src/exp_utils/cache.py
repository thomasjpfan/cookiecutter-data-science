import os
from functools import wraps

import pandas as pd


def from_dataframe_cache(key):
    def cache_decor(f):
        @wraps(f)
        def wrapper(config, force=False, **kwargs):
            fn = config['files'][key]
            if os.path.exists(fn) and not force:
                return pd.read_parquet(fn)
            output = f(config, force, **kwargs)
            output.to_parquet(fn)
            return output
        return wrapper
    return cache_decor
