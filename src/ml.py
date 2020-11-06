import pandas as pd
from src.io import load_jsonls
import numpy as np

def process_raw_data():
    dataset = load_jsonls('raw')
    dataset = pd.DataFrame(dataset)
    dataset.drop(['headline-raw', 'claps-raw'], axis=1, inplace=True)

    dupes_mask = dataset.duplicated()
    print(f'dropping {dupes_mask.sum()} duplicates')
    dataset = dataset.loc[~dupes_mask, :]
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
    return dataset


if __name__ == '__main__':
    dataset = process_raw_data()
    dataset = create_features(dataset)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder

    from sklearn.compose import ColumnTransformer

    from sklearn.pipeline import Pipeline

    from sklearn.preprocessing import FunctionTransformer
    words_pipe = Pipeline([
        ('headline-tidf', TfidfVectorizer()),
        ('todense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
    ])

    transformer = ColumnTransformer(
        [
            ('headline-tidf', words_pipe, 'headline'),
            ('one-hot', OneHotEncoder(handle_unknown='error'), ['site_id', ])
        ],
        remainder='drop'
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import FeatureUnion
    pipe = Pipeline([
        ('features', transformer),
        ('est', GaussianNB())
    ])

    pipe.fit(dataset, dataset['binned-class'])
    steps = transformer.named_transformers_

    for name, step in steps.items():
        try:
            n_features = step.get_feature_names()
            print(f'{name}, {len(n_features)} cols')
        except AttributeError:
            pass

    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    preds = pipe.predict(dataset)
    score = balanced_accuracy_score(dataset['binned-class'], preds)

    cli = 'Pokemon Masters Gems Hack'

    cli = pd.DataFrame({
        'headline': cli,
        'claps': -1,
        'site_id': 'hacker-daily',
        'year': -1,
        'site': 'hacker-daily'
    }, index=[0])
    cli = create_features(cli)

    print(pipe.predict(cli))
