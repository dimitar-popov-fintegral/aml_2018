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
def cluster_tester(n,k):

    ##
    rs = numpy.random.RandomState(12357)

    ##
    logger.info('Data prep')
    x_train = rs.randint(100, size=(n, k))
    y_train = rs.randint(3,   size=(n, 1))

    learning_rate = numpy.logspace(-2, -0.5, 10)
    param_grid = dict(learning_rate=learning_rate)
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    grid_search = GridSearchCV(model, param_grid, scoring="balanced_accuracy", n_jobs=4, cv=kfold)
    opt_ada_boost_params = grid_search.fit(x_train, y_train.flatten())
    logger.info("Best: [{:f}] using [{}]".format(opt_ada_boost_params.best_score_, opt_ada_boost_params.best_params_))

    return opt_ada_boost_params


#######################################################################
if __name__ == '__main__':

    root = logging.getLogger(__name__)
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch) 

    ##
    user = 'dpopov@student.ethz.ch'
    logger = logging.getLogger(__name__)
    opt_params = cluster_tester(n=100, k=10)
    logger.info('{} <Job Done>'.format(user))