import pandas as pd
from src.io import load_jsonls
import numpy as np

def process_raw_data(year_start=2014):
    dataset = load_jsonls('raw')
    dataset = pd.DataFrame(dataset)
    dataset.drop(['headline-raw', 'claps-raw'], axis=1, inplace=True)

    dupes_mask = dataset.duplicated('headline')
    print(f'dropping {dupes_mask.sum()} duplicates')
    dataset = dataset.loc[~dupes_mask, :]

    dataset['year'] = dataset['year'].astype(int)
    year_mask = dataset.loc[:, 'year'] >= year_start
    n = year_mask.shape[0] - year_mask.sum()
    print(f'dropping {n} rows for {year_start} year start')
    dataset = dataset.loc[year_mask, :]
    assert dataset['year'].min() >= year_start

    print(f'ds shape: {dataset.shape}')
    return dataset


from sklearn.preprocessing import KBinsDiscretizer
def create_binned_target(dataset):
    return KBinsDiscretizer(
        n_bins=4,
        strategy='quantile',
        encode='ordinal'
    )


def create_features(
    dataset
):
    binner = create_binned_target(dataset)
    dataset.loc[:, 'binned-class'] = binner.fit_transform(dataset['claps'].values.reshape(-1, 1))

    dataset.loc[:, 'log-claps'] = np.log(dataset['claps'])
    dataset.loc[:, 'clip-claps'] = np.clip(
        dataset['claps'],
        a_min=0,
        a_max=80000
    )
    dataset.loc[:, 'n-characters'] = dataset.loc[:, 'headline'].apply(lambda x: len(x))
    dataset.loc[:, 'n-words'] = dataset.loc[:, 'headline'].apply(lambda x: len(x.split(' ')))
    return dataset
from sklearn.base import BaseEstimator, TransformerMixin

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


if __name__ == '__main__':
    dataset = process_raw_data(year_start=2017)
    dataset = create_features(dataset)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder

    from sklearn.compose import ColumnTransformer

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import FunctionTransformer
    words_pipe = Pipeline([
        ('tidf', TfidfVectorizer()),
        ('todense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
    ])

    transformer = ColumnTransformer(
        [
            ('words', words_pipe, 'headline'),
            ('one-hot', OneHotEncoder(handle_unknown='ignore'), ['site_id', ]),
            ('target-encoding', TargetEncoding(), ['site_id', 'claps']),
            ('n-characters', 'passthrough', ['n-characters'],),
            ('n-words', 'passthrough', ['n-words'],)
        ],
        remainder='drop'
    )
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
    from sklearn.pipeline import FeatureUnion
    pipe = Pipeline([
        ('features', transformer),
        ('est', MultinomialNB())
    ])
    from sklearn.ensemble import RandomForestClassifier
    parameters = [
        {
            'est': [MultinomialNB()],
            'features__words__tidf__stop_words': [None],
            'features__target-encoding': ['drop', TargetEncoding()]
        },
        {
            'est': [GaussianNB()],
            'features__words__tidf__stop_words': [None],
            'features__target-encoding': ['drop', TargetEncoding()]
        },
        {
            'est': [RandomForestClassifier()]
            'features__target-encoding': ['drop', TargetEncoding()]
        }
    ]
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(pipe, parameters)

    from sklearn.model_selection import train_test_split
    x_tr, x_te, y_tr, y_te = train_test_split(dataset, dataset['binned-class'])

    gs.fit(x_tr, y_tr)
    print(gs.cv_results_['mean_test_score'])

    import pdb; pdb.set_trace()



    pipe.fit(x_tr, y_tr)
    steps = transformer.named_transformers_

    for name, step in steps.items():
        try:
            n_features = step.get_feature_names()
            print(f'{name}, {len(n_features)} cols')
        except AttributeError:
            pass

    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    pred_tr = pipe.predict(x_tr)
    score_tr = balanced_accuracy_score(y_tr, pred_tr)
    pred_te = pipe.predict(x_te)
    score_te = balanced_accuracy_score(y_te, pred_te)
    print(score_tr, score_te)

    cli = 'Top Ten Python Tips'
    print(cli)

    cli = pd.DataFrame({
        'headline': cli,
        'claps': -1,
        'site_id': 'personal-growth',
        'year': -1,
        'site': 'hacker-daily'
    }, index=[0])
    cli = create_features(cli)

    print(pipe.predict(cli))

    #  check predictions / accuracy by site
    #  TODO error analysis of worst predictions (use the probability from the naive bayes model)
    #  do this in a notebook :)

    #  ordinal encoding of the site by median claps in the training data?

    #  expts were
    #  over stop words -> not remove
    #  over classifires -> bayse is fine


