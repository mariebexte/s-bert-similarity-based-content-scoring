import pandas as pd
from pathlib import Path
import sys

path = Path(sys.argv[1])

for results_path in path.iterdir():
    df = pd.read_csv(results_path/'scores.csv')
    mean_vals = df.groupby(['dataset', 'model', 'features']).mean().drop(columns='fold')
    mean_model_vals = df.groupby(['dataset', 'model']).mean().drop(columns='fold')
    mean_feature_vals = df.groupby(['dataset', 'features']).mean().drop(columns='fold')
    mean_datasets = df.groupby('dataset').mean().drop(columns='fold')

    names = ['scores_means.csv', 'scores_means_models.csv', 'scores_means_features.csv', 'scores_means_datasets.csv']
    dfs = [mean_vals, mean_model_vals, mean_feature_vals, mean_datasets]

    for fn, dfm in zip(names, dfs):
        dfm.to_csv(results_path/fn)
