import json
from pathlib import Path

import pandas as pd

from src.dirs import DATAHOME

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
