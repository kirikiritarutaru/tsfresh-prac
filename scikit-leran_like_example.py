import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter

warnings.simplefilter('ignore')


def sklearn_like_classifier_pipeline():
    df_ts, y = load_robot_execution_failures()
    df_ts.set_index(['id', 'time'], inplace=True)

    pipeline = Pipeline([
        ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
        ('classifier', RandomForestClassifier()),
    ])

    # 'id'のインデックスを抽出し、学習データのidとテストデータのidに分割
    unique_ids = df_ts.index.get_level_values('id').unique()
    X_train, X_test, y_train, y_test = train_test_split(unique_ids, y, test_size=0.3, random_state=42, stratify=y)

    # 時系列データ準備（tsfreshの処理向けにindexを戻す）
    train_df = df_ts[df_ts.index.get_level_values('id').isin(X_train)].reset_index()
    test_df = df_ts[df_ts.index.get_level_values('id').isin(X_test)].reset_index()

    X = pd.DataFrame(index=y_train.index)
    pipeline.set_params(augmenter__timeseries_container=train_df)
    pipeline.fit(X, y_train)
    y_pred = pipeline.predict(X)

    print('Train:')
    print(classification_report(y_train, y_pred))

    pipeline.set_params(augmenter__timeseries_container=test_df)
    X = pd.DataFrame(index=y_test.index)
    y_pred = pipeline.predict(X)

    print('Test:')
    print(classification_report(y_test, y_pred))  # 問題が簡単すぎる？


if __name__ == '__main__':
    sklearn_like_classifier_pipeline()
