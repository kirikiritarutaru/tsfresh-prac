from pprint import pprint

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.examples.robot_execution_failures import (
    download_robot_execution_failures, load_robot_execution_failures)
from tsfresh.utilities.dataframe_functions import impute


def load_robot_data():
    download_robot_execution_failures()
    ts, y = load_robot_execution_failures()

    return ts, y


def check_data(ts, num=3):
    ts.set_index(['id', 'time'], inplace=True)
    col = ts.columns.tolist()
    colors = cm.viridis(np.linspace(0, 1, len(col)))

    fig, axs = plt.subplots(len(col), len(range(1, num + 1)), sharex=True, figsize=(24, 10))
    for i, feature in enumerate(col):
        for j, id in enumerate(range(1, num + 1)):
            ts.xs(id, level='id')[feature].plot(ax=axs[i, j], color=colors[i])
            axs[i, j].set_title(f'id: {id}, {feature}')
    plt.tight_layout()
    plt.show()


def ex_feat(ts):
    extracted_features = extract_features(ts, column_id='id', column_sort='time')
    return extracted_features


def ex_rel_feat(ef, y):
    impute(ef)
    features_filtered = select_features(ef, y)
    return features_filtered


if __name__ == '__main__':
    ts, y = load_robot_data()
    print(f'id counts: {len(ts["id"].unique())}')
    # check_data(ts, num=3)

    ef = ex_feat(ts)
    pprint(ef)
    ff = ex_rel_feat(ef, y)
    pprint(ff)
