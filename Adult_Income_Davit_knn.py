#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
main_df = pd.concat([train_df, test_df])
X_dev = main_df.drop('income', axis=1)
y_dev = main_df['income']
scaler = MinMaxScaler()
X_dev = scaler.fit_transform(X_dev)


class RFIFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_features_=10):
        self.n_features_ = n_features_
        self.fs_indices_ = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from numpy import argsort
        model_rfi = RandomForestClassifier(n_estimators=100)
        model_rfi.fit(X, y)
        self.fs_indices_ = argsort(model_rfi.feature_importances_)[
            ::-1][0:self.n_features_]
        return self

    def transform(self, X, y=None):
        return X[:, self.fs_indices_]


def get_search_results(gs):

    def model_result(scores, params):
        scores = {'mean_score': np.mean(scores),
                  'std_score': np.std(scores),
                  'min_score': np.min(scores),
                  'max_score': np.max(scores)}
        return pd.Series({**params, **scores})

    models = []
    scores = []

    for i in range(gs.n_splits_):
        key = f"split{i}_test_score"
        r = gs.cv_results_[key]
        scores.append(r.reshape(-1, 1))

    all_scores = np.hstack(scores)
    for p, s in zip(gs.cv_results_['params'], all_scores):
        models.append((model_result(s, p)))

    pipe_results = pd.concat(models, axis=1).T.sort_values(
        ['mean_score'], ascending=False)

    columns_first = ['mean_score', 'std_score', 'max_score', 'min_score']
    columns = columns_first + \
        [c for c in pipe_results.columns if c not in columns_first]

    return pipe_results[columns]


def run_knn():

    n_samples = 20000

    Data_sample = pd.DataFrame(X_dev).sample(
        n=n_samples, random_state=8).values
    target_sample = pd.DataFrame(y_dev).sample(
        n=n_samples, random_state=8).values
    Data_sample_train, Data_sample_test, target_sample_train, target_sample_test = train_test_split(Data_sample, target_sample,
                                                                                                    test_size=0.3, random_state=999,
                                                                                                    stratify=target_sample)
    cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)

    pipe_KNN = Pipeline(steps=[('rfi_fs', RFIFeatureSelector()),
                               ('knn', KNeighborsClassifier())])

    params_pipe_KNN = {'rfi_fs__n_features_': [10, 20, X_dev.shape[1]],
                       'knn__n_neighbors': [1, 5, 10, 15, 20],
                       'knn__p': [1, 2]}

    gs_pipe_KNN = GridSearchCV(estimator=pipe_KNN,
                               param_grid=params_pipe_KNN,
                               cv=cv_method,
                               refit=True,
                               n_jobs=-2,
                               scoring='roc_auc',
                               verbose=1)

    t_start = time.time()
    gs_pipe_KNN.fit(Data_sample_train, target_sample_train.ravel())
    t_end = time.time()

    results_KNN = get_search_results(gs_pipe_KNN)
    results_KNN.head()

    cv_method_ttest = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=111)
    cv_results_KNN = cross_val_score(estimator=gs_pipe_KNN.best_estimator_,
                                     X=Data_sample_test,
                                     y=target_sample_test.ravel(),
                                     cv=cv_method_ttest,
                                     n_jobs=-2,
                                     scoring='roc_auc')

    p_start = time.time()
    pred_KNN = gs_pipe_KNN.predict(Data_sample_test)
    p_end = time.time()
    report = metrics.classification_report(
        target_sample_test, pred_KNN,  output_dict=True)
    f1_score = report['macro avg']['f1-score']

    print(f"KNN train time = {t_end - t_start}")
    print(f"KNN prediction time = {p_end - p_start}")
    print(f"cv_results_KNN.mean() = {cv_results_KNN.mean()}")
    print(f"f1-score for knn = {f1_score}")


run_knn()
