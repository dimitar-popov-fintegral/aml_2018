import os
import sys

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import logging
import numpy
import pandas
import logging 

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix as confusion_matrix
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import data as dt 


#######################################################################
def train_svm_classifier(x_train, y_train, x_test, y_test, machines, 
                        c_penalty_lower, c_penalty_upper, c_penalty_num, 
                        g_lower, g_upper, g_num, class_weight, kernel, comment='SMVC'):

    ##
    logger = logging.getLogger(__name__)
    rs = numpy.random.RandomState(12357)


    ##
    support_vector_machine_classifier = SVC(gamma='scale', kernel=kernel, class_weight=class_weight)

    ##
    logger.info('<--Spec model parameters-->')
    if (g_lower == None) & (g_upper == None) & (g_num == None):
        gamma = 'scale'
        c_penalty = numpy.logspace(c_penalty_lower, c_penalty_upper, c_penalty_num)
        param_grid = dict(C=c_penalty)
    else:
        c_penalty = numpy.logspace(c_penalty_lower, c_penalty_upper, c_penalty_num)
        gamma = numpy.logspace(g_lower, g_upper, g_num)
        param_grid = dict(C=c_penalty, gamma=gamma)

    kfold = StratifiedKFold(n_splits=3, random_state=rs)
    score = make_scorer(f1_score, average='micro')
    

    ##
    logger.info('<--Start grid search over C and gamma-->')
    grid_search = GridSearchCV(support_vector_machine_classifier, param_grid, scoring=score, n_jobs=machines, cv=kfold, verbose=3)
    opt_model = grid_search.fit(x_train, y_train.values.flatten())
    logger.info("Best score: [{:f}] using [{}]".format(opt_model.best_score_, opt_model.best_params_))

    ##
    logger.info('<--Make prediction and write out-->')
    prediction = opt_model.predict(x_test)
    prediction_score = f1_score(prediction, y_test.values.flatten(), average='micro')
    logger.info('Check prediction score on validation set := [{:f}]'.format(prediction_score))

    output = pandas.Series(prediction, name='y')
    output.to_csv(os.path.join(dt.output_dir(), 'SVM_{:s}.csv'.format(comment)), index=True, header=['y'], index_label=['id'])
    print(confusion_matrix(prediction, y_test.values.flatten(), labels=list(range(len(set(y_test.y.values))))))

    return prediction, opt_model

