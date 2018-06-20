from functools import wraps

import pandas as pd


def from_dataframe_cache(key):
    def cache_decor(f):
        @wraps(f)
        def wrapper(params, force=False, **kwargs):
            fn = params[key]
            if fn.exists() and not force:
                return pd.read_parquet(fn)
            output = f(params, force, **kwargs)
            output.to_parquet(fn)
            return output
        return wrapper
    return cache_decor
