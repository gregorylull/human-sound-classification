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

# (glull) custom functions
from src.utilities import data as gldata

def reduce_dimensions(X, var_explained=0.5, prev_pca = None):
    pca = prev_pca or PCA(var_explained)
    X_transformed = pca.fit_transform(X)

    return (pca, X_transformed)

# this function was for debugging, not actually used
def single_test(train_val_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(train_val_test['X_cv'])
    y_train = train_val_test['y_cv']

    X_test = train_val_test['X_test']
    X_test_scaled = scaler.transform(X_test)
    y_test = train_val_test['y_test']

    print('\ndata set train balance: ', sum(y_train)/len(y_train) * 100)
    print('data set test_ balance: ', sum(y_test)/len(y_test) * 100)

    # single inner loop
    pca, X_train_pca = reduce_dimensions(X_train_scaled, 0.5)
    logregcv = LogisticRegressionCV(Cs=[1000, 5000, 10000], cv=3)
    logregcv.fit(X_train_pca, y_train)

    # y_predictions = logregcv.predict(X_test_scaled, y_test)
    X_test_transformed = pca.transform(X_test_scaled)
    model_score = logregcv.score(X_test_transformed, y_test)

    print('\ntrain score:', logregcv.score(X_train_pca, y_train))
    print('test_ score:', logregcv.score(X_test_transformed, y_test))

    return (logregcv, train_val_test, scaler, pca)

def train_fit_no_pipeline(train_val_test):
    """
    (glull)
    DO NOT USE.
    Unfortunately because logisticregressionCV does not have a param that
    controls how X_train is scaled before it is split (at least that i know of),
    the train set will be (could be?) contaminated with the Validation fold.
    
    This is left here as reference for myself.
    
    Alternatives:
    - use LogisticRegression with kfold
    - use Pipelines with GridSearchCV
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(train_val_test['X_cv'])
    y_train = train_val_test['y_cv']

    X_test = train_val_test['X_test']
    X_test_scaled = scaler.transform(X_test)
    y_test = train_val_test['y_test']

    print('\ndata set train balance: ', sum(y_train)/len(y_train) * 100)
    print('data set test_ balance: ', sum(y_test)/len(y_test) * 100)

    results = []
    for var_explained in np.arange(0.1, 0.9, 0.1):

        pca, X_train_pca = reduce_dimensions(X_train_scaled, var_explained)
        logregcv = LogisticRegressionCV(Cs=np.arange(1, 10000, 100), cv=3, max_iter=1000)
        logregcv.fit(X_train_pca, y_train)

        X_test_transformed = pca.transform(X_test_scaled)
        y_pred = logregcv.predict(X_test_transformed)
        train_model_score = logregcv.score(X_train_pca, y_train)
        test_model_score = logregcv.score(X_test_transformed, y_test)
        report = classification_report(y_test, y_pred, target_names=['no_human', 'has_human'], output_dict=True)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'var_explained': var_explained,
            'score_metric': f1,
            'test_model_default_score': test_model_score,
            'train_model_default_score': train_model_score,
            'n_components': pca.n_components_,
            'logregcv_cs': logregcv.C_,
            'pca': pca,
            'scaler': scaler,
            'f1_no_human': report['no_human']['f1-score'],
            'f1_has_human': report['has_human']['f1-score'],
            'f1_score': f1

        })
    
    results_sorted = sorted(results, reverse=True, key=lambda obj: obj['score_metric'])
    return results_sorted

