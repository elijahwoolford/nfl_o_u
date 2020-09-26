import operator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

import model_pipeline


def get_NN_hyperparameters(X_train, y_train, nn_model):
    parameter_space = {
        'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,),
                               (15,), (16,), (17,), (18,), (19,), (20,), (21,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'learning_rate': ['constant', 'adaptive'],
    }

    clf = GridSearchCV(nn_model, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    print("-" * 80)
    print('Best parameters found:\n', clf.best_params_)
    print("-" * 80)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print("-" * 80)

    return clf.best_params_


def get_svm_hyperparameters(X_train, y_train, svm_model):
    parameter_space = {'C': [0.1, 1, 10, 100, 1000],
                       'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                       'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    clf = RandomizedSearchCV(svm_model, parameter_space, n_jobs=-1, cv=5, n_iter=5)
    clf.fit(X_train, y_train)
    print("-" * 80)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print("-" * 80)
    print('Best parameters found:\n', clf.best_params_)
    print("-" * 80)

    return clf.best_params_


def get_rf_hyperparameters(X_train, y_train, rf_model):
    pass


