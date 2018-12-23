"""Text Model"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

from runner import get_runner
import fire

categories = [
    'alt.atheism',
    'talk.religion.misc',
]


def predict(run):

    test_data = fetch_20newsgroups(subset='test', categories=categories)
    text_model = joblib.load(run.cfg.text__model_fn)
    y_predict = text_model.predict(test_data.data)
    np.save(run.cfg.text__prediction_fn, y_predict)
    run.log.info(f"Finished prediction: {run.model_id}")


def train_hp(run):

    from dask.distributed import Client
    from dask_ml.model_selection import GridSearchCV
    # client = Client('localhost:8786')
    client = Client()
    workers = client.ncores()
    run.log.info(f"Dask distributed: Connected to {workers}")

    data = fetch_20newsgroups(subset='train', categories=categories)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(max_iter=1000)),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        # 'clf__alpha': (0.00001, 0.000001),
        # 'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
    }
    grid_search = GridSearchCV(pipeline, parameters, scheduler=client)
    grid_search.fit(data.data, data.target)
    train_score = grid_search.best_score_

    test_data = fetch_20newsgroups(subset='test', categories=categories)
    test_score = grid_search.score(test_data.data, test_data.target)
    best_params = grid_search.best_params_

    joblib.dump(grid_search.best_estimator_, run.cfg.text__model_fn)

    run.log.info(
        f"Finished hyperparameter search model_id: {run.model_id}, test_score: "
        f"{test_score:0.6}, train_score: {train_score:0.6}, "
        f"params: {best_params}")

    return {"valid": test_score, "train": train_score}


if __name__ == '__main__':
    r = get_runner("text", [train_hp, predict])
    fire.Fire(r)
