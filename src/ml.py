import json

from joblib import dump, load
import pandas as pd
import numpy as np

from nltk.stem.snowball import EnglishStemmer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from src.dirs import MODELHOME, DATAHOME
from src.data import process_raw_data, create_features


def to_dense(x):
    return x.todense()


def process_grid_search_results(gs):
    cv_res = gs.cv_results_
    res = pd.concat([
        pd.DataFrame(cv_res['params']),
        pd.DataFrame(cv_res['mean_test_score'], columns=['test']),
        pd.DataFrame(cv_res['mean_train_score'], columns=['train']),
        pd.DataFrame(cv_res['std_test_score'], columns=['test-std'])
    ], axis=1)
    res.sort_values('test', ascending=False, inplace=True)
    res.columns = [c.split('__')[-1] for c in res.columns]
    return res


def evaluate_model(est, x_tr, y_tr, x_ho, y_ho):
    pred_tr = est.predict(x_tr)
    score_tr = balanced_accuracy_score(y_tr, pred_tr)
    pred_ho = est.predict(x_ho)
    score_ho = balanced_accuracy_score(y_ho, pred_ho)
    return {'score-tr': score_tr, 'score-ho': score_ho}


def train_final_models(params, pipe, dataset, path):
    """trains on both training & entire dataset"""
    path = MODELHOME / path
    path.mkdir(exist_ok=True)
    print(f'final model: {params}')
    ds = dataset

    pipe.set_params(**params)
    pipe.fit(ds['x_tr'], ds['y_tr'])
    dump(pipe, path / 'pipe-tr.joblib')

    score_ho = evaluate_model(
        gs2.best_estimator_,
        ds['x_tr'], ds['y_tr'], ds['x_ho'], ds['y_ho']
    )
    with open(path / 'holdout-score.json', 'w') as fi:
        json.dump(score_ho, fi)

    pipe.fit(ds['x'], ds['y'])
    dump(pipe, path / 'pipe-fi.joblib')

    for name, data in ds.items():
        data.to_csv((path / name).with_suffix('.csv'))

    with open(path / 'params.json', 'w') as fi:
        json.dump({name: repr(o) for name, o in params.items()}, fi)


class TargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, target='claps', group='site_id'):
        self.target = target
        self.group = group

    def fit(self, x, y=None):
        grps = x.groupby(self.group).mean().loc[:, self.target].to_frame()
        self.grps = grps.to_dict()[self.target]

        grps = x.groupby(self.group).count().loc[:, self.target].to_frame()
        self.freq_grps = grps.to_dict()[self.target]
        return self

    def transform(self, x):
        t = x.loc[:, self.group].replace(self.grps).to_frame()
        f = x.loc[:, self.group].replace(self.freq_grps).to_frame()
        return pd.concat([t, f], axis=1)

    def get_feature_names(self):
        return ['target-mean', 'target-freq']


st = EnglishStemmer()
analyzer = TfidfVectorizer().build_analyzer()
def stemmer(doc):
    return (st.stem(w) for w in analyzer(doc))


def gridsearch(
    pipe,
    parameters,
    name
):
    print(f'\nstarting {name}')
    gs = GridSearchCV(
         pipe,
         parameters,
         n_jobs=6,
         scoring=scorer,
         refit=True,
         return_train_score=True,
         verbose=True
     )
    gs.fit(x_tr, y_tr)
    res = process_grid_search_results(gs)
    res.to_csv(MODELHOME / name)


if __name__ == '__main__':
    sub = 50
    dataset = process_raw_data(year_start=2017)
    dataset = dataset.iloc[:sub, :]
    dataset, binner = create_features(dataset)

    dump(binner, DATAHOME / 'processed' / 'binner.joblib')

    words_pipe = Pipeline([
        ('tidf', TfidfVectorizer(analyzer=stemmer, stop_words='english')),
        ('todense', FunctionTransformer(to_dense, accept_sparse=True)),
    ])

    transformer = ColumnTransformer(
        [
            ('words', words_pipe, 'headline'),
            ('one-hot', OneHotEncoder(handle_unknown='ignore'), ['site_id', ]),
            ('target-encoding', TargetEncoding(), ['site_id', 'claps']),
            ('month', 'passthrough', ['month']),
            ('n-characters', 'passthrough', ['n-characters']),
            ('n-words', 'passthrough', ['n-words'])
        ],
        remainder='drop'
    )
    pipe = Pipeline([
        ('features', transformer),
        ('est', GaussianNB())
    ])

    scorer = make_scorer(balanced_accuracy_score)
    x = dataset
    y = dataset['binned-class']
    x_tr, x_ho, y_tr, y_ho = train_test_split(x, y, shuffle=True)

    dataset = {
        'x_tr': x_tr,
        'y_tr': y_tr,
        'x_ho': x_ho,
        'y_ho': y_ho,
        'x': x,
        'y': y
    }

    gs1 = [
        {
            'est': [GaussianNB()],
            'features__words__tidf__stop_words': [None],
            'features__words__tidf__analyzer': ['word', stemmer],
            'features__target-encoding': [TargetEncoding()]
        },
        {
            'est': [RandomForestClassifier(n_estimators=100)],
            'features__words__tidf__stop_words': [None],
            'features__words__tidf__analyzer': ['word', stemmer],
            'features__target-encoding': [TargetEncoding()]
        },
        {
            'est': [GradientBoostingClassifier(n_estimators=100)],
            'features__words__tidf__stop_words': [None],
            'features__words__tidf__analyzer': ['word', stemmer],
            'features__target-encoding': ['drop', TargetEncoding()]
        }
    ]
    gridsearch(pipe, gs1, 'gridsearch1')

    gs2 = [
        {
            'est': [RandomForestClassifier()],
            'est__n_estimators': [100, 250],
            'est__max_depth': [10, None],
            'est__max_features': ['sqrt', 'log2'],
            'features__words__tidf__stop_words': ['english'],
            'features__target-encoding': [TargetEncoding()]
        }
    ]
    gridsearch(pipe, gs2, 'gridsearch2')
    train_final_models(gs2.best_params_, pipe, dataset, 'rf')

    gs3 = [
        {
            'est': [GradientBoostingClassifier()],
            'est__n_estimators': [250, 500, 1000],
            'est__max_depth': [2, 3, 4],
            'est__learning_rate': [0.05, 0.01],
            'est__max_features': ['sqrt'],
            'features__words__tidf__stop_words': ['english'],
            'features__target-encoding': [TargetEncoding()]
        }
    ]
    gridsearch(pipe, gs3, 'gridsearch3')
    train_final_models(gs3.best_params_, pipe, dataset, 'gb')

    nb_params = {
        'est': GaussianNB(),
        'features__words__tidf__stop_words': 'english',
        'features__target-encoding': 'drop'
    }
    train_final_models(nb_params, pipe, dataset, 'naive')
