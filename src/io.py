import json
import os
from pathlib import Path

from joblib import load
import pandas as pd

from src.dirs import HOME, DATAHOME, MODELHOME


def load_csvs(folder='raw', recursive=False):
    """loads all CSVs in a directory"""
    base = Path(HOME, 'data', folder)

    if recursive:
        pattern = '**/*.csv'
    else:
        pattern = '*.csv'

    data = {}
    for fpath in base.glob(pattern):
        #  in case we hit CSVs pandas can't parse, ignore ParserError
        try:
            df = pd.read_csv(fpath, low_memory=False)
            #  drop an index col if we load one by accident
            df.drop("Unnamed: 0", axis=1, inplace=True, errors='ignore')
            data[str(fpath.relative_to(base))] = df
        except pd.errors.ParserError:
            pass
    return data


def load_jsonls(folder='raw', recursive=False):
    """loads all CSVs in a directory"""
    base = Path(DATAHOME, folder)
    pattern = '**/*.jsonl'
    dataset = []
    for fpath in base.glob(pattern):
        with open(fpath, 'r') as fi:
            for line in fi.readlines():
                data = json.loads(line)
                dataset.append(data)
    return dataset


def load_artifacts(fldr):
    """load model artifacts such as csvs, model and JSON files"""
    path = MODELHOME / fldr
    artifacts = {
        'name': fldr, 'path': str(path)
    }
    for csv in path.glob('*.csv'):
        artifacts[csv.stem] = pd.read_csv(csv, index_col=0)

    for pipe in path.glob('*.joblib'):
        artifacts[pipe.stem] = load(pipe)

    for js in path.glob('*.json'):
        with open(js, 'r') as fi:
            artifacts[js.stem] = json.load(fi)

    return artifacts
