"""Text Model"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from dask_ml.model_selection import GridSearchCV
import joblib
import numpy as np

from utils import generate_experiment_params_from_env, normalize_params

exp, params, n_ctx = generate_experiment_params_from_env("text", tags=["text"])

categories = [
    'alt.atheism',
    'talk.religion.misc',
]


@exp.command
def predict(model_id, run_dir, _log):
    p = normalize_params(params, run_dir)

    test_data = fetch_20newsgroups(subset='test', categories=categories)
    text_model = joblib.load(p.text__model_fn)
    y_predict = text_model.predict(test_data.data)
    np.save(p.text__prediction_fn, y_predict)
    _log.info(f"Finished prediction, model_id: {model_id}")


@exp.command
def train_hp(model_id, run_dir, _log, _run):
    p = normalize_params(params, run_dir)

    from dask.distributed import Client
    client = Client('192.168.2.34:8786')

    data = fetch_20newsgroups(subset='train', categories=categories)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(max_iter=1000)),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
    }
    grid_search = GridSearchCV(pipeline, parameters, scheduler=client)
    grid_search.fit(data.data, data.target)
    train_score = grid_search.best_score_

    test_data = fetch_20newsgroups(subset='test', categories=categories)
    test_score = grid_search.score(test_data.data, test_data.target)
    best_params = grid_search.best_params_

    joblib.dump(grid_search.best_estimator_, p.text__model_fn)

    _log.warning(
        f"Finished hyperparameter search model_id: {model_id}, test_score: "
        f"{test_score:0.6}, train_score: {train_score:0.6}, "
        f"params: {best_params}")

    return [test_score, train_score]


@exp.command
def train(model_id, run_dir, _log, _run):
    _log.info(f"No normal training use train_hp")
