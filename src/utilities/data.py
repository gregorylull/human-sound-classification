import glob
import pandas as pd
import numpy as np
import re
import json
import pickle
from matplotlib import pylab as plt

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, f1_score, make_scorer, accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

# At this point i do not know how to get the root folder programmatically,
# if this file/folder is moved around the rel path needs to be updated
ROOT_FOLDER = '../../'
CHIME_FOLDER = f'analysis/chime/'

def get_params(model='logreg', **kwargs):

    # exit early and use model defaults when testing pipeline and introducing new models.
    if 'fast' in kwargs and kwargs['fast']:
        print(f'\n  Fast modeling for {model}\n')
        return {
            'pca__n_components': [1]
        }

    # there could be thousands of features, so instead of narrowing down
    # how many components to capture, this is indicating how much variation to capture.
    # 0.1 means keep enough pca components to explain at least 10% of the variation.
    pca_component_range = np.arange(0.1, 1.0, 0.1)

    if model=='logreg':
        return {
            # the higher the C the more important the features.
            # the lower the C the more powerful the regularization.
            'logreg__C': np.arange(1, 1000, 100),
            'pca__n_components': pca_component_range
        }

    elif model=='svm':
        return {
            'svm__kernel': ['linear', 'poly', 'rbf'],

            # the higher the gamma the more likely model will overfit to the data.
            'svm__gamma': [*np.arange(0.1, 1, 0.2), 1, 3, 5, 10],

            # C relates to the penalty term, moves the boundary, high will lead to overfit
            'svm__C': [0.1, 0.5, 1, 3, 5, 10],

            # degree is for polynomial
            'svm__degree': [1, 2, 3, 4, 5],

            'pca__n_components': pca_component_range
        }
    
    elif model=='knn':
        return {
            'knn__n_neighbors': np.arange(5, 30, 5),
            'pca__n_components': pca_component_range
        }
    
    elif model=='randomforest':
        return {
            'pca__n_components': pca_component_range,
            'randomforest__n_estimators': np.arange(10, 150, 15)
        }


def get_grid_pipeline(model_type, params_=None):
    def pipeline(X, y, **kwargs):

        if params_:
            params = params_
        else:
            params = get_params(model_type, **kwargs) 

        grid = train_fit_test_pipeline(
            X,
            y,
            model_type,
            params,
            **kwargs
        )

        return grid

    return pipeline


def get_model(model='logreg'):
    if model=='logreg':
        return LogisticRegression(solver='lbfgs', max_iter=1000)

    elif model=='svm':
        return svm.SVC(gamma='auto', cache_size=1000, max_iter=5000)

    elif model=='knn':
        return KNeighborsClassifier()
    
    elif model=='randomforest':
        return RandomForestClassifier(n_estimators=10)

# returns scorer(estimator, X_, y_)
def get_scoring_metric(**kwargs):
    metric = kwargs['metric'] if 'metric' in kwargs and kwargs['metric'] else 'accuracy'

    if metric == 'f1':
        return make_scorer(f1_score, greater_is_better=True)

    elif metric == 'accuracy':
        return make_scorer(accuracy_score, greater_is_better=True)

def get_pipeline(model_type):

    model = get_model(model_type)

    model_pipeline = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('pca', PCA()),
        (model_type, model)
    ])

    return model_pipeline


def train_fit_test_pipeline(X, y, model_type='logreg', params={}, cv=3, **kwargs):
    """
    using pipeline to prevent crossfold leakage, e.g.:
    https://towardsdatascience.com/pre-process-data-with-pipeline-to-prevent-data-leakage-during-cross-validation-e3442cca7fdc
    """
    model_pipeline = get_pipeline(model_type)

    scoring_metric = get_scoring_metric(**kwargs)

    grid = GridSearchCV(model_pipeline, params, cv=cv, scoring=scoring_metric)

    grid.fit(X, y)

    return grid

def split(X, y, test_size=0.2):
    """
    Split data into stratified Train, Validate, Test.
    """


    # get Test, set aside
    X_remainder, X_test, y_remainder, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )

    # get Train, Validate
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_remainder, y_remainder, test_size=test_size, stratify=y_remainder
    )

    return {
        # original
        'X': X,
        'y': y,

        # train test split
        'X_train': X_train,
        'y_train': y_train,
        'X_validate': X_validate,
        'y_validate': y_validate,
        'X_test': X_test,
        'y_test': y_test,

        # this is from the first split, can use cross validation
        'X_cv': X_remainder, 
        'y_cv': y_remainder
    }

def get_train_val_test(use_saved=False):
    """
    This is specific to the chime data set
    """
    tvt_filename = f'{CHIME_FOLDER}train_val_test.pkl'

    print('\nTrainValTest - using cached pkl :', use_saved)

    if use_saved:
        with open(tvt_filename, 'rb') as readfile:
            train_val_test = pickle.load(readfile)
    else:

        # clean and remove NaN
        chime_mfcc_filename = f'{CHIME_FOLDER}chime_mfcc.csv'
        raw_df = pd.read_csv(chime_mfcc_filename)
        df = raw_df.dropna(axis='columns')

        # get features and target
        X = df.drop(columns=['Unnamed: 0', 'has_child', 'has_male', 'has_female', 'has_human', 'chunkname'])
        y = df['has_human']
        train_val_test = split(X, y)

        with open(tvt_filename, 'wb') as writefile:
            pickle.dump(train_val_test, writefile)

    print('\nloading train_val_test pkl for chime:\n', train_val_test.keys(),'\n')
    for key, val in train_val_test.items():
        print(key, val.shape)

    return train_val_test

class NaiveModel:

    def __init__(self, y, predicting_class=1):
        self.results = [predicting_class] * len(y)

    def predict(self, *args, **kwargs):
        return self.results
