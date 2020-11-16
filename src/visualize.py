import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importances(rf, cols, n=50):
    importances = pd.DataFrame()
    importances.loc[:, 'importances'] = rf.feature_importances_
    importances.loc[:, 'features'] = cols
    importances.sort_values('importances', inplace=True, ascending=False)
    importances = importances.iloc[:n, :]
    f, a = plt.subplots()
    importances.plot(ax=a, kind='bar', x='features', y='importances')
    plt.gcf().subplots_adjust(bottom=0.3)
    return f, a
