import os
import logging
import numpy
from enum import Enum

from scipy import stats
import statsmodels.api as sm
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

VALIDATION_SET_SIZE = 0.2
SEED = 12357
BALANCE=[0.125, 0.75, 0.125]

"""
Task2 Notes:
    - Dealing with class imbalance 
        1.- Consider different measures of performance 
            a. confusion matrix
            b. precision
            d. recall
            e. Cohen's Kappa
            f. RoC curves
        2.- Over- and under-sampling of the data
            a. over-sampling by creating copies of under-represented data
            b. under-sampling by deleting over-represented data
            c. The way in which you sample is also important, try stratafied random sampling
            d. Python 'UnbalancedDataset' 
        3.- Semi-supervised learning?   
"""

#######################################################################
class Tasks(Enum):

    TASK0 = 'task0'
    TASK1 = 'task1'
    TASK2 = 'task2'


#######################################################################
def data_dir():

    return os.path.join(THIS_DIR, 'store')


#######################################################################
def output_dir():

    if not os.path.isdir(os.path.join(THIS_DIR, 'output')):
        os.mkdir(os.path.join(THIS_DIR, 'output'), mode=0o770)

    return os.path.join(THIS_DIR, 'output')


#######################################################################
def create_validation_set(y_train, validation_set_size=VALIDATION_SET_SIZE, seed=SEED, imbalance=False, enforce_imbalance_ratio=True):

    logger = logging.getLogger(__name__)

    ##
    RandomState = numpy.random.RandomState(seed)
    n=int(len(y_train.index) * validation_set_size)


    if imbalance:
        classes = y_train.y.unique()
        logger.info('proposed class splits [{}]'.format(classes))
        counts = numpy.array([(y_train.y == i).sum() for i in classes])
        ratios = counts / len(y_train.y)
        num_samples = numpy.floor(n * ratios).astype(int)

        logger.info('sampling for [{}] classes, in proportions [{}]'.format(len(classes), ratios))

        idx = numpy.zeros(0).astype(int)
        for i, k in zip(classes, num_samples):
            mask = y_train.y == i
            sample = y_train.y[mask].sample(n=k, random_state=RandomState).index
            idx = numpy.append(idx, sample.astype(int))

        assert numpy.unique(idx).__len__() == len(idx),\
            'sampling did not result in unique index'

        sample_classes = y_train.y.reindex(idx).unique()
        sample_counts = numpy.array([(y_train.y.reindex(idx) == i).sum() for i in sample_classes])
        sample_ratios = sample_counts / len(y_train.y.reindex(idx))

        if enforce_imbalance_ratio:
            assert numpy.testing.assert_allclose(ratios, sample_ratios, rtol=1e-1) is None,\
                'sampling produced inaccurate sample class proportions [{}] \n\
                in comparison to train class proportions [{}]'.format(sample_ratios, ratios)

    else:
        idx = y_train.y.sample(n=n, random_state=RandomState).index.values.astype(int)

    return idx


#######################################################################
def small_variance_cols(x_train, threshold):

    cols_std = x_train.describe().transpose()['std']
    return cols_std[cols_std<threshold].index


#######################################################################
def remove_outliers(x_train, zscore_threshold):

    for p in x_train.columns:
        predictor = x_train[p]
        predictor[abs(stats.zscore(predictor.fillna(predictor.mean()))) > zscore_threshold] = np.nan


#######################################################################
def fill_missing_with_ols(x_train, prior_data):

    data = pd.DataFrame(index=x_train.index)

    for p in x_train.columns:
        print(p)

        predictor = x_train[p]

        missing_idx = predictor[predictor.isnull()].index
        good_idx = predictor.index.difference(missing_idx)

        X = prior_data
        X = X.fillna(X.mean())

        #X = (X-X.mean()/X.std())
        lm = sm.OLS(predictor.reindex(good_idx).values,
                    sm.add_constant(X.reindex(good_idx).values)).fit()

        missing = pd.Series(name=p, index=missing_idx,
                            data=lm.predict(sm.add_constant(X.reindex(missing_idx).values)))
        predictor = pd.concat([predictor.reindex(good_idx), missing]).reindex(data.index)

        predictor = (predictor-predictor.mean())/predictor.std()
        assert not predictor.isnull().values.any()
        assert all(predictor.index == data.index)
        data = data.join(predictor, how='outer')

    return data



#######################################################################
def write_validation_set(x_train, y_train, suffix):

    index = create_validation_set(y_train)
    x_train.reindex(index).to_csv(os.path.join(output_dir(), 'x_{}.csv'.format(suffix)), index=True)
    y_train.reindex(index).to_csv(os.path.join(output_dir(), 'y_{}.csv'.format(suffix)), index=True)

    pass