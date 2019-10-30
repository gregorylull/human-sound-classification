import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix

# my code
from src.utilities import data as gldata

ROOT_FOLDER = '../../'
CURRENT_FOLDER = f'analysis/chime/'

USE_CACHED_TRAIN_VAL_TEST = True

METRIC = 'f1'

# TODO this is so messy, folder structure needs to be reorganized
# and have a single source of truth
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
    'logreg': get_filename('grid_logreg'),
    'svm': get_filename('grid_svm'),
    'knn': get_filename('grid_knn'),
    'randomforest': get_filename('grid_randomforest'),

    # final results
    'final_results': get_filename('model_final_results')
}

# roc
# example1: https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/
# example2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
def create_roc_curve(y, predicted_probs, model_type):

    fpr, tpr, thresholds = roc_curve(y, predicted_probs)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr,
        color='darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with {model_type}')
    plt.legend(loc="lower right")

    return plt

# plotting curve example: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def create_pca_cumulative_curve(pca, title):
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')

    if title:
        plt.title(title)

    return plt

def save_plot(plt, filename):
    params = dict(
        figsize=[16, 9],
        dpi=300,
        bbox_inches='tight',
    )
    plt.savefig(get_filename(f'{filename}', '.svg'), **params)
    plt.savefig(get_filename(f'{filename}', '.png'), **params)

def get_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tn, fp, fn, tp

def main():
    train_val_test = gldata.get_train_val_test(USE_CACHED_TRAIN_VAL_TEST)

    # look at chime_analysis.py for results contract
    with open(PICKLE_FILENAMES['final_results'], 'rb') as readfile:
        final_results = pickle.load(readfile)
    
    model_type = final_results['final_model_type']
    model_pipeline = final_results['final_model_pipeline']

    # create ROC curve
    # this is a binary classification, just want the y=1 column
    predicted_proba = model_pipeline.predict_proba(train_val_test['X_test'])
    y = train_val_test['y_test'].tolist()
    y_pred = predicted_proba[:, 1].tolist()
    roc_plt = create_roc_curve(y, y_pred, model_type)
    save_plot(roc_plt, f'{model_type}_roc')

    # create PCA curve for model
    grid = final_results['model_comparisons'][0]['model']
    pca = grid.best_estimator_.named_steps['pca']
    pca_plt = create_pca_cumulative_curve(pca, f'PCA cumulative curve with {model_type}')
    save_plot(pca_plt, f'{model_type}_pca_cumulative')

    # create PCA curve without model
    pca_all = PCA()
    X_transformed = StandardScaler().fit_transform(train_val_test['X'])
    pca_all.fit(X_transformed)
    pca_all_plt = create_pca_cumulative_curve(pca_all, f'PCA cumulative curve')
    save_plot(pca_all_plt, f'pca_all_cumulative')

    # get confusion matrix
    tn, fp, fn, tp = get_confusion_matrix(train_val_test['y_test'], final_results['final_predict'])
    print('\ntrue neg', tn)
    print('false pos', fp)
    print('false neg', fn)
    print('true pos', tp)


    return final_results


if __name__ == '__main__':
    results = main()
