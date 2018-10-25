import os
import sys

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy
import pandas
import logging 

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import matplotlib.pyplot as plt 
import data as dt

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
def ada_boost_experiment(x_train, y_train, x_test, y_test, x_submit, base_classifier, max_depth, 
    n_estimators, learning_rate_lower, learning_rate_upper, learning_rate_num, comment='AdaBoostClassifier'):

    logger = logging.getLogger(__name__)
    rs = numpy.random.RandomState(12357)

    ##
    learning_rate = numpy.logspace(learning_rate_lower, learning_rate_upper, learning_rate_num)
    model = AdaBoostClassifier(base_classifier(max_depth=max_depth))
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=rs)
    grid_search = GridSearchCV(model, param_grid, scoring="balanced_accuracy", n_jobs=48, cv=kfold, verbose=3)
    opt_ada_boost_params = grid_search.fit(x_train, y_train.values.flatten())
    logger.info("Best: [{:f}] using [{}]".format(opt_ada_boost_params.best_score_, opt_ada_boost_params.best_params_))

    check_score = balanced_accuracy_score(opt_ada_boost_params.predict(x_test), y_test.values.flatten())
    logger.info('Check prediction score on validation set := [{:f}]'.format(check_score))

    ##
    logger.info('Refit AdaBoostClassifier w/ best params from CV')
    ada_boost_classifier = AdaBoostClassifier(
        base_classifier, 
        n_estimators=opt_ada_boost_params.best_params_['n_estimators'], 
        learning_rate=opt_ada_boost_params.best_params_['learning_rate']
    )
    
    ##
    ada_boost_classifier.fit(x_train, y_train.values.flatten())
    score = balanced_accuracy_score(ada_boost_classifier.predict(x_test), y_test.values.flatten())
    logger.info('Prediction score on validation set := [{:f}]'.format(score))

    ada_boost_classifier_test_accuracy = []
    for ada_boost_classifier_test_predict in ada_boost_classifier.staged_predict(x_test):
        ada_boost_classifier_test_accuracy.append(balanced_accuracy_score(
            y_true=ada_boost_classifier_test_predict, 
            y_pred=y_test.y.values.flatten()))

    y_submit = ada_boost_classifier.predict(x_submit)
    output = pandas.Series(y_submit, name='y')
    output.to_csv(os.path.join(dt.output_dir(), 'AdaBoost_{:s}.csv'.format(comment)), index=True, header=['y'], index_label=['id'])

    y_check_submit = opt_ada_boost_params.predict(x_submit)
    output = pandas.Series(y_check_submit, name='y')
    output.to_csv(os.path.join(dt.output_dir(), 'AdaCheck_{:s}.csv'.format(comment)), index=True, header=['y'], index_label=['id'])

    # Boosting might terminate early, but the following arrays are always
    # n_estimators long. We crop them to the actual number of trees here:
    ada_boost_classifier_num_trees = len(ada_boost_classifier)
    ada_boost_classifier_estimator_errors = ada_boost_classifier.estimator_errors_[:ada_boost_classifier_num_trees]

    ##
    logger.info('plotting results')
    plt.ioff()
    fig = plt.figure(figsize=(15, 15))
    plt.plot(
        range(1, ada_boost_classifier_num_trees + 1),
         ada_boost_classifier_test_accuracy, c='black',
         linestyle='dashed', label='AdaBoostClassifier with SAMME.R algo.'
    )
    plt.legend()
    plt.ylabel('Balanced Accuracy Score')
    plt.xlabel('Number of Trees')
    plt.savefig(os.path.join(dt.output_dir(), 'AdaBoostEvol_{:s}.png'.format(comment)))
    plt.close(fig)    


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
        base_classifier = DecisionTreeClassifier(),
        x_train=x_train, 
        y_train=y_train, 
        x_test=x_test, 
        y_test=y_test,
        x_submit=x_submit,
        max_depth=3,     
        n_estimators = [800, 1000, 1500, 2000],
        learning_rate_lower = -1,
        learning_rate_upper = 0.5,
        learning_rate_num = 15,
    )

    args_to_report = [
        'max_depth',
        'n_estimators',
        'learning_rate_lower',
        'learning_rate_upper',
        'learning_rate_num',
    ]

    comment_kwargs = {key: classifier_kwargs[key] for key in args_to_report}

    comment = 'params_{}'.format(comment_kwargs).replace(' ', '').replace("'","")
    logger.info('Running AdaBoostClassifier w/ parameters defined by: \n [{:s}]'.format(comment))
    ada_boost_experiment(**classifier_kwargs, comment=comment)

