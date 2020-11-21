from joblib import load
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.svm import LinearSVC

from src.dirs import MODELHOME, DATAHOME
from src.ml import to_dense, TargetEncoding, stemmer
from src.ml import evaluate_model, train_final_models
from src.io import load_artifacts
from src.data import create_features


if __name__ == '__main__':
    print('Loading models...')
    mdls = [d for d in MODELHOME.glob('*') if not d.is_file()]
    mdls = {path.name: load_artifacts(path) for path in mdls}
    print(f'{mdls.keys()} loaded')
    import warnings
    warnings.filterwarnings("ignore")
    while True:
        # inp = 'Animated Guide to Deep Learning Layers'
        print('Input headline:')
        inp = input()

        cli = pd.DataFrame({
            'headline': inp,
            'claps': -1,
            'site_id': 'towardsdatascience',
            'year': -1,
            'month': 12,
            'site': 'towardsdatascience.com',
        }, index=[0])
        cli, _ = create_features(cli)

        binner = load(DATAHOME / 'processed' / 'binner.joblib')
        edges = binner.bin_edges_

        edges = edges[0].flatten().tolist()
        starts = edges[:-1]
        ends = edges[1:]

        probs = mdls['rf']['pipe-fi'].predict_proba(cli).flatten().tolist()
        pred = mdls['rf']['pipe-fi'].predict(cli)

        print(f'prediction is {pred}')
        for label, (start, end, prob) in enumerate(zip(starts, ends, probs)):
                print(f'label {label} start {start:6.0f}, end {end:6.0f}, prob {100*prob:3.1f}%')
