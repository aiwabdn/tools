#!/usr/bin/env python

from __future__ import print_function

import importlib
import json

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PipelineGridSearch:
    def __init__(self, model_name, params, pipeline=None):
        if pipeline is None:
            self.pipeline = Pipeline(
                [('scaler', StandardScaler()), ('model', None)])
        elif 'model' not in pipeline.named_steps:
            raise ValueError(
                'You must provide a `model` step in your pipeline. That should be the name of any model you plan to use'
            )
        else:
            self.pipeline = pipeline
        self.params = {}
        for k, v in params.items():
            self.params['model__{}'.format(k)] = v

        model_index = -1
        for idx, i in enumerate(self.pipeline.steps):
            if i[0] == 'model':
                model_index = idx

        module_name = model_name[:model_name.rfind('.')]
        module = importlib.import_module(module_name)
        model = getattr(module, model_name.split('.')[-1])
        self.pipeline.steps[model_index] = ('model', model())

    def fit(self, X, y, **kwargs):
        gridsearch = GridSearchCV(self.pipeline, self.params, **kwargs)
        gridsearch.fit(X, y)
        self.best_params = gridsearch.best_params_
        self.best_estimator = gridsearch.best_estimator_
        self.best_score = gridsearch.best_score_

    def predict(self, X):
        return self.best_estimator.predict(X)

    def score(self, X, y):
        return self.best_estimator.score(X, y)


class ModelSelector:
    def __init__(self, params, pipeline=None):
        self.params = params
        self.pipeline = pipeline
        self.best_model_name = None
        self.best_model = None
        self.best_params = None
        self.all_best_models = {}
        self.all_best_params = {}
        self.all_best_scores = {}

    def fit(self, X, y, **kwargs):
        best_score = -10000000
        for model_name, params in self.params.items():
            print("Training {}".format(model_name))
            pipe_grid = PipelineGridSearch(model_name, params, self.pipeline)
            pipe_grid.fit(X, y, **kwargs)
            self.all_best_models[model_name] = pipe_grid.best_estimator
            self.all_best_scores[model_name] = pipe_grid.best_score
            self.all_best_params[model_name] = pipe_grid.best_params
            if pipe_grid.best_score > best_score:
                print('Setting best model to {}'.format(model_name))
                best_score = pipe_grid.best_score
                self.best_model_name = model_name
                self.best_model = pipe_grid.best_estimator
                self.best_params = pipe_grid.best_params

    @property
    def best_model(self):
        return self.best_model

    @property
    def best_params(self):
        return self.best_params

    def all_best_predict(self, X):
        preds = {}
        for k, v in self.all_best_models.items():
            preds[k] = v.predict(X)

        return preds

    def all_best_score(self, X, y):
        scores = {}
        for k, v in self.all_best_models.items():
            scores[k] = v.score(X, y)

        return scores

    def all_best_score(self, X, y):
        scores = {}
        for k, v in self.all_best_models.items():
            scores[k] = v.score(X, y)

        return scores

    def get_best_stats(self, X, y):
        pred = self.best_model.predict(X)
        conf_mat = pd.DataFrame(
            confusion_matrix(y, pred),
            index=np.unique(y),
            columns=np.unique(y))
        prfs_mat = pd.DataFrame(
            precision_recall_fscore_support(y, pred),
            index=['precision', 'recall', 'fscore', 'support'],
            columns=np.unique(y))
        return conf_mat, prfs_mat

    def get_all_best_stats(self, X, y):
        preds = self.all_best_predict(X)
        conf_mats = {}
        prfs_mats = {}
        for k, v in preds.items():
            conf_mats[k] = pd.DataFrame(
                confusion_matrix(y, v),
                index=np.unique(y),
                columns=np.unique(y))
            prfs_mats[k] = pd.DataFrame(
                precision_recall_fscore_support(y, v),
                index=['precision', 'recall', 'fscore', 'support'],
                columns=np.unique(y))

        return conf_mats, prfs_mats


# sample parameters
sample_hyperparameters = {
    'xgboost.XGBClassifier': {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': range(1, 6),
        'n_estimators': range(100, 600, 50),
        'reg_alpha': [0, 0.2, 0.5, 1.0]
    },
    'sklearn.linear_model.SGDClassifier': {
        'loss': ['hinge', 'squared_hinge', 'log'],
        'penalty': ['l1', 'l2'],
        'shuffle': [True],
        'learning_rate': ['optimal', 'adaptive'],
        'eta0': [0.1, 0.01, 0.05]
    },
    'sklearn.svm.SVC': {
        'C': [0.1, 0.5, 1.0, 2.0],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'degree': [3, 5, 7],
        'tol': [1e-3, 1e-5],
        'gamma': ['auto', 0.01, 0.005]
    }
}

# # demo
# # create a model selector
# ms = ModelSelector(hyperparameters, pipeline)

# # fit the selector to the data, the api is the same as GridSearchCV
# ms.fit(X_train, y_train, cv=3, verbose=2)

# # might take a looong time depending on how big a parameter grid you have specified

# # print the best params
# print(ms.best_params)

# # store the best model
# dump(ms.best_model, '/workdir/outputs/best_model.joblib')

# # get the stats of the best model found in the training
# confusion_mat, prfs_mat = ms.get_best_stats(X_test, y_test)

# # you can also dump the stats from all the best models found in training
# confusion_mats, prfs_mats = ms.get_all_best_stats(X_test, y_test)

# for k, v in confusion_mats.items():
#     v.to_csv('/workdir/outputs/confusion_mats.txt', sep='\t', mode='a')
#     with open('/workdir/outputs/confusion_mats.txt', 'a') as f:
#         f.write('\n')
#         f.write(k)
#         f.write('\n' + '-' * 50 + '\n\n')

# for k, v in prfs_mats.items():
#     v.to_csv('/workdir/outputs/prfs_mats.txt', sep='\t', mode='a')
#     with open('/workdir/outputs/prfs_mats.txt', 'a') as f:
#         f.write('\n')
#         f.write(k)
#         f.write('\n' + '-' * 50 + '\n\n')
