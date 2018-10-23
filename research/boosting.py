import numpy
import pandas
import logging 
import os 
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import matplotlib.pyplot as plt 

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


#######################################################################
def train_ada_boost_classifier(x_train, y_train, x_test, y_test, base_classifier, n_estimators, learning_rate):
    pass


#######################################################################
def ada_boost_experiment(x_train, y_train, x_test, y_test, x_submit, base_classifier, 
    n_estimators, learning_rate_lower, learning_rate_upper, learning_rate_num, comment='AdaBoostClassifier'):

    logger = logging.getLogger(__name__)

    learning_rate = numpy.logspace(learning_rate_lower, learning_rate_upper, learning_rate_num)
    model = AdaBoostClassifier(base_classifier, n_estimators=n_estimators)
    param_grid = dict(learning_rate=learning_rate)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="balanced_accuracy", n_jobs=-1, cv=kfold)
    opt_ada_boost_params = grid_search.fit(x_train, y_train.values.flatten())
    logger.info("Best: [{:f}] using [{}]".format(opt_ada_boost_params.best_score_, opt_ada_boost_params.best_params_))
    breakpoint()

    ##
    logger.info('Refit AdaBoostClassifier w/ best params from 5-fold CV')
    ada_boost_classifier = AdaBoostClassifier(
        base_classifier, 
        n_estimators=n_estimators, 
        learning_rate=opt_ada_boost_params.best_params_['learning_rate']
    )

    ada_boost_classifier_test_accuracy = []
    for ada_boost_classifier_test_predict in ada_boost_classifier.staged_predict(x_test):
        ada_boost_classifier_test_accuracy.append(balanced_accuracy_score(
            y_true=ada_boost_classifier_test_predict, 
            y_pred=y_test.y.values.flatten()))

    # Boosting might terminate early, but the following arrays are always
    # n_estimators long. We crop them to the actual number of trees here:
    ada_boost_classifier_num_trees = len(ada_boost_classifier)
    ada_boost_classifier_estimator_errors = ada_boost_classifier.estimator_errors_[:ada_boost_classifier_num_trees]

    ##
    logger.info('plotting results')
    plt.figure(figsize=(15, 15))
    plt.plot(
        range(1, ada_boost_classifier_num_trees + 1),
         ada_boost_classifier_test_accuracy, c='black',
         linestyle='dashed', label='AdaBoostClassifier with SAMME.R algo.'
    )
    plt.legend()
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees')
    plt.show()


#######################################################################
if __name__ == '__main__':

    root = logging.getLogger(__name__)
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch) 

    from task2 import read_data 
    from data import create_validation_set

    ##
    logger = logging.getLogger(__name__)

    ##
    x_submit, x_train, y_train, _, _  = read_data()
    
    idx_oos_test = create_validation_set(
        y_train=y_train, 
        imbalance=True
    )

    x_test = x_train.reindex(idx_oos_test)
    y_test = y_train.reindex(idx_oos_test)

    x_train.drop(idx_oos_test, inplace=True)
    y_train.drop(idx_oos_test, inplace=True)

    classifier_kwargs = dict(
        base_classifier = DecisionTreeClassifier(max_depth=2),
        x_train=x_train, 
        y_train=y_train, 
        x_test=x_test, 
        y_test=y_test,
        x_submit=x_submit,     
        n_estimators = 600,
        learning_rate_lower = -3,
        learning_rate_upper = -0.5,
        learning_rate_num = 5,
    )

    args_to_report = [
        'n_estimators',
        'learning_rate_lower',
        'learning_rate_upper',
        'learning_rate_num',
    ]

    comment_kwargs = {key: classifier_kwargs[key] for key in args_to_report}

    comment = 'AdaBoostClasifer_params_{}'.format(comment_kwargs)
    logger.info('Running AdaBoostClassifier w/ parameters defined by: \n [{:s}]'.format(comment))
    ada_boost_experiment(**classifier_kwargs, comment=comment)

