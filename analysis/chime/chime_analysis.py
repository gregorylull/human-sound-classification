
"""
(glull)
The aim for this file is to do all the analysis for the CHIME project, including:
- train test split
- standardScale features
- PCA for dimension reduction
- Upsample for a more balanced dataset (currently it is about 35:65 1:0)
- Fit models:
    - LogReg
    - RandomForest

TODO

MUST
- Need to use Validation to check model accuracy that is using different
  number of features (PCA components), DO NOT USE just remainder.
  - done, using Validation to check between classifiers

- LogisticRegressionCV can't guarantee your X input is going to be split
  and standardized correctly. DO NOT USE.
  - done, replaced with pipeline + gridsearchcv

NICE TO HAVE
- Play around with thresholds for classifying
    - is there a way to add this to the pipeline itself, e.g., given a certain combo
      of coefs_, i want to see the metric scores using different proba ranges

- stratified sampling for my dataset, I think this just needs to be added to my get_train_val_test method?


"""
# At this point i do not know how to get the root folder programmatically,
# if this file/folder is moved around the rel path needs to be updated
from analysis.chime import reference_code as glref
from src.utilities import data as gldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pylab as plt
import pickle
import json
import re
import numpy as np
import pandas as pd
import glob
ROOT_FOLDER = '../../'
CURRENT_FOLDER = f'analysis/chime/'


# (glull) custom functions

# Set to true to use pre split test val train sets.
USE_CACHED_TRAIN_VAL_TEST = True

# Set to true to use defaults for all the models, used primarily to:
# 1) test pipeline, and 2) introduce new models
FAST_MODELING = False

# Metric to compare gridCV, options: 'f1' | 'accuracy'
METRIC = 'f1'

# Set to False to retrain to model, otherwise use pkl.
# USE_CACHED = {
#     'logreg': False,
#     'svm': False,
#     'knn': False
# }

# debug
USE_CACHED = {
    'logreg': False,
    'svm': False,
    'knn': False,
    'randomforest': False
}

CONFIG = {
    'metric': METRIC
}


def get_filename(name, ext='.pkl', **kwargs):
    folder = CURRENT_FOLDER
    context = CONFIG.copy()
    context.update(kwargs)

    items = (([val for key, val in context.items()]))
    params_postfix = '_'.join(sorted(items))

    return f'{folder}{name}_{params_postfix}{ext}'


PICKLE_FILENAMES = {
    'logreg': get_filename('logreg'),
    'svm': get_filename('svm'),
    'knn': get_filename('knn'),
    'randomforest': get_filename('randomforest'),

    # final results
    'final_results': get_filename('model_final_results')
}

MODEL_PIPELINES = {
    'logreg': gldata.get_grid_pipeline('logreg'),
    'svm': gldata.get_grid_pipeline('svm'),
    'knn': gldata.get_grid_pipeline('knn'),
    'randomforest': gldata.get_grid_pipeline('randomforest'),
}


def test_models(X, y):
    model_grids = {}

    for model_type, model in MODEL_PIPELINES.items():
        print(f'\nTesting {model_type}')

        # Use cached models when:
        # 1) testing the pipeline
        # 2) introducing new models
        if USE_CACHED[model_type]:
            print(f'  Using cached model for {model_type}\n')
            with open(PICKLE_FILENAMES[model_type], 'rb') as readfile:
                model_grids[model_type] = pickle.load(readfile)

        else:
            model_grids[model_type] = model(
                X, y, metric=METRIC, fast=FAST_MODELING)
            if not FAST_MODELING:
                print(f'  Saving model for {model_type}\n')
                with open(PICKLE_FILENAMES[model_type], 'wb') as writefile:
                    pickle.dump(model_grids[model_type], writefile)

    check_model_grids = model_grids
    return model_grids


def compare_models(X, y, model_grids):
    print('\nComparing models\n')

    scores = []

    for model_type, grid in model_grids.items():
        score = grid.score(X, y)
        scores.append({
            'model_type': model_type,
            'score_metric': score,
            'model': grid
        })

    sorted_scores = sorted(
        scores, key=lambda obj: obj['score_metric'], reverse=True)

    return sorted_scores


def main():
    train_val_test = gldata.get_train_val_test(USE_CACHED_TRAIN_VAL_TEST)

    # use pipeline + grid to test models and then compare models based on a f1 score metric
    model_grids = test_models(
        train_val_test['X_train'], train_val_test['y_train'])
    comparisons = compare_models(
        train_val_test['X_validate'], train_val_test['y_validate'], model_grids)

    # based on comparisons get the best model and its optimized params
    top_model_type = comparisons[0]['model_type']
    top_model = comparisons[0]['model']
    top_model_params = top_model.best_params_

    # create the final model and fit to the X_cv (X_train + X_validate) with:
    # 1) a pipeline because of scale and PCA transformation
    # 2) the best params from the top scoring model
    final_model_pipeline = gldata.get_pipeline(top_model_type)
    final_model_pipeline.set_params(**top_model_params)
    final_model_pipeline.fit(train_val_test['X_cv'], train_val_test['y_cv'])

    scorer = gldata.get_scoring_metric(metric=METRIC)
    final_predict = final_model_pipeline.predict(train_val_test['X_test'])
    final_score = scorer(
        final_model_pipeline,
        train_val_test['X_test'],
        train_val_test['y_test']
    )

    naive_model_score = scorer(
        gldata.NaiveModel(train_val_test['y_test']),
        train_val_test['X_test'],
        train_val_test['y_test']
    )

    final_results = {
        'score_metric': METRIC,
        'scorer': scorer,
        'model_comparisons': comparisons,
        'final_model_type': top_model_type,
        'final_model_pipeline': final_model_pipeline,
        'final_score': final_score,
        'final_predict': final_predict,
        'naive_model_score': naive_model_score
    }

    if not FAST_MODELING:
        print(f'\nSaving final results\n')
        with open(PICKLE_FILENAMES['final_results'], 'wb') as writefile:
            pickle.dump(final_results, writefile)

    print('\nmain() completed\n')
    print('\nnaive_score:', final_results['naive_model_score'])
    print('\nfinal_score:', final_results['final_score'])
    print('\n(final - naive) / naive:',
          (final_results['final_score'] - final_results['naive_model_score'])/final_results['naive_model_score'] * 100, '%')
    print('\nfinal_model_type:', final_results['final_model_type'])

    return final_results


if __name__ == '__main__':
    results = main()

# {'var_explained': 0.5,
#   'metric': 0.8487179487179487,
#   'n_components': 79,
#   'logregcv_cs': array([1]),
#   'pca': PCA(copy=True, iterated_power='auto', n_components=0.5, random_state=None,
#       svd_solver='auto', tol=0.0, whiten=False),
#   'scaler': StandardScaler(copy=True, with_mean=True, with_std=True)}
