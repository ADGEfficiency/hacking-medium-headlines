import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

load_dotenv()
home = os.getenv('PROJECT_HOME')


def load_csvs(folder='raw', recursive=False):
    """loads all CSVs in a directory"""
    base = Path(home, 'data', folder)

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
