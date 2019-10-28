
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
from sklearn.pipeline import Pipeline

# (glull) custom functions
from src.utilities import data as gldata

USE_CACHED_TRAIN_VAL_TEST = True

def get_train_val_test(use_saved):
    """
    This is specific to the chime data set
    """
    tvt_filename = f'{CURRENT_FOLDER}train_val_test.pkl'

    print('\nTrainValTest - using cached pkl :', use_saved)

    if use_saved:
        with open(tvt_filename, 'rb') as readfile:
            train_val_test = pickle.load(readfile)
    else:

        # clean and remove NaN
        chime_mfcc_filename = f'{CURRENT_FOLDER}chime_mfcc.csv'
        raw_df = pd.read_csv(chime_mfcc_filename)
        df = raw_df.dropna(axis='columns')

        # get features and target
        X = df.drop(columns=['Unnamed: 0', 'has_child', 'has_male', 'has_female', 'has_human', 'chunkname'])
        y = df['has_human']
        train_val_test = gldata.split(X, y)

        with open(tvt_filename, 'wb') as writefile:
            pickle.dump(train_val_test, writefile)

    print('\nloading train_val_test pkl for chime:\n', train_val_test.keys(),'\n')
    for key, val in train_val_test.items():
        print(key, val.shape)

    return train_val_test

def reduce_dimensions(X, var_explained=0.5, prev_pca = None):
    pca = prev_pca or PCA(var_explained)
    X_transformed = pca.fit_transform(X)

    return (pca, X_transformed)


def main():
    train_val_test = get_train_val_test(USE_CACHED_TRAIN_VAL_TEST)
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(train_val_test['X_cv'])
    y_train = train_val_test['y_cv']

    X_test = train_val_test['X_test']
    X_test_scaled = scaler.transform(X_test)
    y_test = train_val_test['y_test']

    print('\ndata set train balance: ', sum(y_train)/len(y_train) * 100)
    print('data set test balance: ', sum(y_test)/len(y_test) * 100)

    # single inner loop
    # pca, X_train_pca = reduce_dimensions(X_train_scaled, 0.5)
    # logregcv = LogisticRegressionCV(Cs=[1000, 5000, 10000], cv=3)
    # logregcv.fit(X_train_pca, y_train)

    # # y_predictions = logregcv.predict(X_test_scaled, y_test)
    # X_test_transformed = pca.transform(X_test_scaled)
    # model_score = logregcv.score(X_test_transformed, y_test)

    # print('\ntrain score:', logregcv.score(X_train_pca, y_train))
    # print('test_ score:', logregcv.score(X_test_transformed, y_test))

    # return (logregcv, train_val_test, scaler, pca)


    results = []
    for var_explained in np.arange(0.5, 0.9, 0.1):

        pca, X_train_pca = reduce_dimensions(X_train_scaled, var_explained)
        logregcv = LogisticRegressionCV(Cs=np.arange(100, 10000, 500), cv=3)
        logregcv.fit(X_train_pca, y_train)

        # y_predictions = logregcv.predict(X_test_scaled, y_test)
        X_test_transformed = pca.transform(X_test_scaled)
        model_score = logregcv.score(X_test_transformed, y_test)

        results.append({
            'var_explained': var_explained,
            'metric': model_score,
            'n_components': pca.n_components_,
            'logregcv_cs': logregcv.C_,
            'pca': pca,
            'scaler': scaler

        })

    results_sorted = sorted(results, reverse=True, key=lambda obj: obj['metric'])
    return results_sorted
        




if __name__ == '__main__':
    results = main()
    print('\main() completed\n')
