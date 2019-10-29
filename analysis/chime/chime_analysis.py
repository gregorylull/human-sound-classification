
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
0. Need to use Validation to check model accuracy that is using different 
  number of features (PCA components), DO NOT USE just remainder.

1. LogisticRegressionCV can't guarantee your X input is going to be split
  and standardized correctly. DO NOT USE.

NICE TO HAVE
2. Play around with thresholds for classifying

"""
# At this point i do not know how to get the root folder programmatically,
# if this file/folder is moved around the rel path needs to be updated
ROOT_FOLDER = '../../'
CURRENT_FOLDER = f'analysis/chime/'

import glob
import pandas as pd
import numpy as np
import re
import json
import pickle
from matplotlib import pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, f1_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


# (glull) custom functions
from src.utilities import data as gldata
from analysis.chime import reference_code as glref

# Set to true to use pre split test val train sets.
USE_CACHED_TRAIN_VAL_TEST = True

# Set to true to use defaults for all the models, used primarily to:
# 1) test pipeline, and 2) introduce new models
FAST_MODELING = False

# Set to False to retrain to model, otherwise use pkl.
USE_CACHED = {
    'logreg': False,
    'svm': False,
    'knn': False
}

PICKLE_FILENAMES = {
    'logreg': f'{CURRENT_FOLDER}grid_logreg.pkl',
    'svm': f'{CURRENT_FOLDER}grid_svm.pkl',
    'knn': f'{CURRENT_FOLDER}grid_knn.pkl',

    # final results
    'final_results': f'{CURRENT_FOLDER}model_final_results.pkl',
}

MODELS = {
    'logreg': LogisticRegression,
    'svm': svm.SVC,
    'knn': KNeighborsClassifier
}

MODEL_PIPELINES = {
    'logreg': gldata.get_grid_pipeline('logreg'),
    'svm': gldata.get_grid_pipeline('svm'),
    'knn': gldata.get_grid_pipeline('knn')
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
            model_grids[model_type] = model(X, y, fast=FAST_MODELING)
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
        
    sorted_scores = sorted(scores, key=lambda obj: obj['score_metric'], reverse=True)
    
    return sorted_scores

def main():
    train_val_test = gldata.get_train_val_test(USE_CACHED_TRAIN_VAL_TEST)

    # use pipeline + grid to test models and then compare models based on a f1 score metric
    model_grids = test_models(train_val_test['X_train'], train_val_test['y_train'])
    comparisons = compare_models(train_val_test['X_validate'], train_val_test['y_validate'], model_grids)

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

    final_predict = final_model_pipeline.predict(train_val_test['X_test'])
    final_score = f1_score(train_val_test['y_test'], final_predict)

    final_results = {
        'model_comparisons': comparisons,
        'final_model_type': top_model_type,
        'final_model_pipeline': final_model_pipeline,
        'final_score': final_score,
        'final_predict': final_predict
    }

    if not FAST_MODELING:
        print(f'\nSaving final results\n')
        with open(PICKLE_FILENAMES['final_results'], 'wb') as writefile:
            pickle.dump(final_results, writefile)

    return final_results


if __name__ == '__main__':
    results = main()
    print('\nmain() completed\n')
    print('\nfinal_score:', results['final_score'])
    print('\nfinal_model_type:', results['final_model_type'])

# {'var_explained': 0.5,
#   'metric': 0.8487179487179487,
#   'n_components': 79,
#   'logregcv_cs': array([1]),
#   'pca': PCA(copy=True, iterated_power='auto', n_components=0.5, random_state=None,
#       svd_solver='auto', tol=0.0, whiten=False),
#   'scaler': StandardScaler(copy=True, with_mean=True, with_std=True)}