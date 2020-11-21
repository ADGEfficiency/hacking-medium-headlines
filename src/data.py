"""cleaning & some feature engineering"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from src.io import load_jsonls


def process_raw_data(year_start=2014, drop_dupes=True):
    dataset = load_jsonls('raw')
    dataset = pd.DataFrame(dataset)
    print(f'raw ds shape: {dataset.shape}')
    dataset.drop(['headline-raw', 'claps-raw'], axis=1, inplace=True)
    dataset.loc[:, 'headline'] = dataset.loc[:, 'headline'].str.lower()

    if drop_dupes:
        dupes_mask = dataset.duplicated('headline')
        print(f'dropping {dupes_mask.sum()} duplicates')
        dataset = dataset.loc[~dupes_mask, :]

    dataset['month'] = dataset['month'].astype(int)
    dataset['year'] = dataset['year'].astype(int)
    year_mask = dataset.loc[:, 'year'] >= year_start
    n = year_mask.shape[0] - year_mask.sum()
    print(f'dropping {n} rows for {year_start} year start')
    dataset = dataset.loc[year_mask, :]
    assert dataset['year'].min() >= year_start

    print(f'processed ds shape: {dataset.shape}')
    return dataset


def create_binned_target(n_bins):
    return KBinsDiscretizer(
        n_bins=n_bins,
        strategy='quantile',
        encode='ordinal'
    )


def create_features(dataset, binner=None, n_bins=5, clip_max=80000):
    if not binner:
        binner = create_binned_target(n_bins)

    dataset.loc[:, 'binned-class'] = binner.fit_transform(dataset['claps'].values.reshape(-1, 1))

    dataset.loc[:, 'log-claps'] = np.log(dataset['claps'])
    dataset.loc[:, 'clip-claps'] = np.clip(
        dataset['claps'],
        a_min=0,
        a_max=clip_max
    )
    dataset.loc[:, 'n-characters'] = dataset.loc[:, 'headline'].apply(lambda x: len(x))
    dataset.loc[:, 'n-words'] = dataset.loc[:, 'headline'].apply(lambda x: len(x.split(' ')))
    return dataset, binner
