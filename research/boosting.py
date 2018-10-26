import os
import sys

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import logging
import numpy
import pandas
import logging 

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import matplotlib.pyplot as plt 
import data as dt

from task2 import read_data 
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#######################################################################
def train_ada_boost_classifier(x_train, y_train, x_test, y_test, max_depth, class_weight, 
                               n_estimators, learning_rate_lower, learning_rate_upper, 
                               learning_rate_num, machines, comment='AdaBoostClassifier'):

    logger = logging.getLogger(__name__)
    rs = numpy.random.RandomState(12357)

    ##
    logger.info('<--Spec model parameters-->')
    learning_rate = numpy.logspace(learning_rate_lower, learning_rate_upper, learning_rate_num)
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight))
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=4, random_state=rs)
    
    ##
    logger.info('<--Start grid search over n_estimators and learning_rate-->')
    grid_search = GridSearchCV(model, param_grid, scoring="balanced_accuracy", n_jobs=machines, cv=kfold, verbose=3)
    opt_model = grid_search.fit(x_train, y_train.values.flatten())
    logger.info("Best score: [{:f}] using [{}]".format(opt_model.best_score_, opt_model.best_params_))

    ##
    logger.info('<--Make prediction and write out-->')
    prediction = opt_model.predict(x_test)
    prediction_score = balanced_accuracy_score(prediction, y_test.values.flatten())
    logger.info('Check prediction score on validation set := [{:f}]'.format(prediction_score))

    output = pandas.Series(prediction, name='y')
    output.to_csv(os.path.join(dt.output_dir(), 'ABC_{:s}.csv'.format(comment)), index=True, header=['y'], index_label=['id'])

    return prediction, opt_model



#######################################################################
def ada_boost_experiment(x_train, y_train, x_test, y_test, x_submit, max_depth, 
    n_estimators, learning_rate_lower, learning_rate_upper, learning_rate_num, comment='AdaBoostClassifier'):

    logger = logging.getLogger(__name__)
    rs = numpy.random.RandomState(12357)

    ##
    learning_rate = numpy.logspace(learning_rate_lower, learning_rate_upper, learning_rate_num)
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth))
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
        DecisionTreeClassifier(max_depth=max_depth), 
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

    y_check_submit = opt_y_check_submitada_boost_params.predict(x_submit)
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
    logger = logging.getLogger(__name__)

    '''
    Round 0: Data read in and validation split
    '''
    x_test, x_train, y_train = read_data()
    
    idx_oos_test = dt.create_validation_set(
        y_train=y_train, 
        imbalance=True
    )

    ##
    explained = 0.99
    scale = StandardScaler()
    scale.fit(x_train)

    pca = PCA(explained, whiten=True)
    pca_x_train = pandas.DataFrame(pca.fit_transform(scale.transform(x_train.values)), index=x_train.index)
    pca_n, pca_f = pca_x_train.shape
    logger.info('number of factors [{:d}] to explain [{:f}] variance '.format(pca_f, explained))    

    '''
    Round 1: Fit without validation set, check score against validation set 
    '''
    
    ##
    logger.debug('STRICTLY MODEL PARAMETERS - COMMON TO FIRST AND SECOND STAGE')
    max_depth = 3
    n_estimators = [400, 600, 800]
    learning_rate_lower = -3
    learning_rate_upper = -1
    learning_rate_num = 20
    machines = 48
    class_weight = 'balanced'
    # 'learning_rate': 2.1983926488622894, 'n_estimators': 1500
    
    classifier_kwargs = dict(
        ## data
        x_train=pca_x_train.drop(index=idx_oos_test), 
        y_train=y_train.drop(idx_oos_test), 
        x_test=pca_x_train.reindex(index=idx_oos_test), 
        y_test=y_train.reindex(idx_oos_test),
        ## params
        max_depth=max_depth,     
        n_estimators=n_estimators,
        learning_rate_lower=learning_rate_lower,
        learning_rate_upper=learning_rate_upper,
        learning_rate_num=learning_rate_num,
        class_weight=class_weight,
        ## aux
        machines = machines
    )

    args_to_report = [
        'max_depth',
        'n_estimators',
        'learning_rate_lower',
        'learning_rate_upper',
        'learning_rate_num',
    ]

    comment_kwargs = {key: classifier_kwargs[key] for key in args_to_report}
    comment = 'first_{}'.format(comment_kwargs).replace(' ', '').replace("'","")

    logger.info('Running First Stage AdaBoostClassifier, parameters defined by: \n\n [{:s} \n\n]'.format(comment))
    prediction, model = train_ada_boost_classifier(**classifier_kwargs, comment=comment)

    ##
    classes = numpy.unique(prediction)
    counts = numpy.array([(prediction == i).sum() for i in classes])
    ratios = counts / len(prediction)
    logger.info('<y-predict (w/o validation data) \n classes: [{}], \n class ratios [{}], \n class counts [{}]> \n'.format(classes, ratios, counts))
    logger.info('Stage one model diagnostic \n\n')
    print(model)

    '''
    Round 2: Fit with validation set, check score against naive classifier i.e. all zeros 
    '''
    classifier_kwargs = dict(
        ## data
        x_train=x_train, 
        y_train=y_train, 
        x_test=x_test, 
        y_test=pandas.Series(0, name='y', index=x_test.index),
        ## params
        max_depth=max_depth,     
        n_estimators = n_estimators,
        learning_rate_lower = learning_rate_lower,
        learning_rate_upper = learning_rate_upper,
        learning_rate_num = learning_rate_num,
        class_weight=class_weight,
        ## aux
        machines = machines
    )

    args_to_report = [
        'max_depth',
        'n_estimators',
        'learning_rate_lower',
        'learning_rate_upper',
        'learning_rate_num',
    ]

    comment_kwargs = {key: classifier_kwargs[key] for key in args_to_report}
    comment = 'second_{}'.format(comment_kwargs).replace(' ', '').replace("'","")

    logger.info('Running Second Stage AdaBoostClassifier, parameters defined by: \n\n [{:s} \n\n]'.format(comment))
    prediction, model = train_ada_boost_classifier(**classifier_kwargs, comment=comment)

    ##
    classes = numpy.unique(prediction)
    counts = numpy.array([(prediction == i).sum() for i in classes])
    ratios = counts / len(prediction)
    logger.info('<y-predict (w/ validation data) \n classes: [{}], \n class ratios [{}], \n class counts [{}]> \n'.format(classes, ratios, counts))
    logger.info('Stage one model diagnostic \n\n')
    print(model)
