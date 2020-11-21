"""gridsearching and final model training"""
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from src.dirs import MODELHOME, DATAHOME
from src.data import process_raw_data, create_features


scorer = make_scorer(balanced_accuracy_score)
st = EnglishStemmer()
analyzer = TfidfVectorizer().build_analyzer()


def stemmer(doc):
    return (st.stem(w) for w in analyzer(doc))


def to_dense(x):
    return x.todense()


def init_pipe():
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
    return pipe


def gridsearch(
    parameters,
    name,
    x_tr, y_tr,
    pipe=None,
    verbose=1,
    n_jobs=6,
    **kwargs

):
    print(f'\nstarting {name}')
    if not pipe:
        pipe = init_pipe()

    gs = GridSearchCV(
         pipe,
         parameters,
         n_jobs=n_jobs,
         scoring=scorer,
         refit=True,
         return_train_score=True,
         verbose=verbose
     )
    gs.fit(x_tr, y_tr)
    res = process_grid_search_results(gs)
    res.loc[:, 'est'] = res.loc[:, 'est'].apply(lambda x: repr(x).split('(')[0])
    res.to_csv((MODELHOME / name).with_suffix('.csv'), index=False)
    return gs


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


def evaluate_model(est, x_tr, y_tr, x_ho, y_ho, **kwargs):
    pred_tr = est.predict(x_tr)
    score_tr = balanced_accuracy_score(y_tr, pred_tr)
    pred_ho = est.predict(x_ho)
    score_ho = balanced_accuracy_score(y_ho, pred_ho)
    return {'score-tr': score_tr, 'score-ho': score_ho}


def train_final_models(
        params, x_tr, y_tr, x_ho, y_ho, x, y, path, pipe=None, **kwargs
):
    """trains on both training & entire dataset"""
    if not pipe:
        pipe = init_pipe()
    path = MODELHOME / path
    path.mkdir(exist_ok=True)

    pipe.set_params(**params)
    pipe.fit(x_tr, y_tr.values.reshape(-1,))
    dump(pipe, path / 'pipe-tr.joblib')

    score_ho = evaluate_model(
        pipe,
        x_tr, y_tr, x_ho, y_ho
    )
    with open(path / 'holdout-score.json', 'w') as fi:
        json.dump(score_ho, fi)

    pipe.fit(x, y.values.reshape(-1,))
    dump(pipe, path / 'pipe-fi.joblib')

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


if __name__ == '__main__':
    sub = -1
    dataset = process_raw_data(year_start=2017)
    dataset = dataset.iloc[:sub, :]
    dataset, binner = create_features(dataset)
    dump(binner, DATAHOME / 'processed' / 'binner.joblib')

    x = dataset
    y = dataset['binned-class']
    x_tr, x_ho, y_tr, y_ho = train_test_split(x, y, shuffle=True)

    dataset = {
        'x_tr': x_tr, 'y_tr': y_tr,
        'x_ho': x_ho, 'y_ho': y_ho,
        'x': x, 'y': y
    }

    data_path = DATAHOME / 'final'
    data_path.mkdir(exist_ok=True)
    for name, data in dataset.items():
        data.to_csv((data_path / name).with_suffix('.csv'))

    gs1 = [
        {
            'est': [GaussianNB()],
            'features__words__tidf__stop_words': [None],
            'features__words__tidf__analyzer': ['word', stemmer],
            'features__target-encoding': [TargetEncoding()],
            'features__one-hot': ['drop'],
            'features__month': ['drop'],
            'features__n-characters': ['drop'],
            'features__n-words': ['drop']
        },
        {
            'est': [MultinomialNB()],
            'features__words__tidf__stop_words': [None],
            'features__words__tidf__analyzer': ['word', stemmer],
            'features__target-encoding': [TargetEncoding()],
            'features__one-hot': [OneHotEncoder(handle_unknown='ignore')],
        },
        {
            'est': [BernoulliNB()],
            'features__words': ['drop'],
            'features__target-encoding': ['drop']
        },
        {
            'est': [BernoulliNB()],
            'features__words': ['drop'],
            'features__target-encoding': ['drop'],
            'features__month': ['drop'],
            'features__n-characters': ['drop'],
            'features__n-words': ['drop']
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
            'features__target-encoding': [TargetEncoding()]
        }
    ]
    gridsearch(gs1, 'gridsearch1', **dataset)

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
    gs2 = gridsearch(gs2, 'gridsearch2', **dataset)
    train_final_models(gs2.best_params_, **dataset, path='rf')

    gs3 = [
        {
            'est': [GradientBoostingClassifier()],
            'est__n_estimators': [100, 300],
            'est__max_depth': [2, 3, 4],
            'est__learning_rate': [0.05, 0.01],
            'est__max_features': ['sqrt'],
            'features__words__tidf__stop_words': ['english'],
            'features__target-encoding': [TargetEncoding()]
        }
    ]
    gs3 = gridsearch(gs3, 'gridsearch3', **dataset)
    train_final_models(gs3.best_params_, **dataset, path='gb')

    naives = [
        ('nb-gau', {
            'est': GaussianNB(),
            'features__words__tidf__stop_words': None,
            'features__words__tidf__analyzer': 'word',
            'features__target-encoding': TargetEncoding(),
            'features__one-hot': 'drop',
            'features__month': 'drop',
            'features__n-characters': 'drop',
            'features__n-words': 'drop'
        }),
        ('nb-multi', {
            'est': MultinomialNB(),
            'features__words__tidf__stop_words': None,
            'features__words__tidf__analyzer': 'word',
            'features__target-encoding': TargetEncoding(),
            'features__one-hot': OneHotEncoder(handle_unknown='ignore'),
        }),
        ('nb-ber', {
            'est': BernoulliNB(),
            'features__words': 'drop',
            'features__target-encoding': 'drop'
        })
    ]
    for name, params in naives:
        train_final_models(params, **dataset, path=name)
