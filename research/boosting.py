import os
import sys

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import logging
import numpy
import pandas
import logging 
import re 

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
def grid_search_analysis(filename):
    
    ##
    logger.debug('Read and prepare cross validation score data')
    grid_search_data = pandas.read_csv(os.path.join(dt.output_dir(), filename), header=None, index_col=None)
    grid_search_data.iloc[:, :-1]
    grid_search_data.columns = ['learning_rate', 'n_estimators', 'score']
    grid_search_data.loc[:, 'score']=grid_search_data.loc[:, 'score'].apply(lambda x: float((re.findall(r'[0.]\d+', x))[0]))
    grid_search_data.loc[:, 'learning_rate']=grid_search_data.loc[:, 'learning_rate'].apply(lambda x: float((re.findall(r'[0.]\d+', x))[0]))
    grid_search_data.loc[:, 'n_estimators']=grid_search_data.loc[:, 'n_estimators'].apply(lambda x: int((re.findall(r'\d+', x))[0]))

    ##
    logger.debug('Return table sorted by score')
    result = grid_search_data.sort_values(by='score')
    print(result)

    ##
    logger.debug('Analytics')
    print(grid_search_data.groupby(('n_estimators', 'learning_rate')).mean().sort_values('score'))


    return result


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
    max_depth = 2
    n_estimators = [400, 600, 800]
    learning_rate_lower = -3
    learning_rate_upper = -2
    learning_rate_num = 20
    machines = 24
    class_weight = 'balanced'
    
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
    if False:
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
