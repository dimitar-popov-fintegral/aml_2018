import os
import sys
import pandas
import numpy
import logging

import data as dt

#######################################################################
def read_data():

    ##
    logger = logging.getLogger(__name__)
    logger.info('Reading data for from [{}]'.format(dt.data_dir(), 'task2'))
    X_test  = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_test.csv'), header=0, index_col=0)
    X_train = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_train.csv'), header=0, index_col=0)
    y_train = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'y_train.csv'), header=0, index_col=0)

    ##
    logger.info('Modify y_train to be of type <<int.>>')
    y_train.y = y_train.y.astype(int)

    return \
    X_test,\
    X_train,\
    y_train,\


#######################################################################
if __name__ == '__main__':

    import matplotlib.pyplot as plt 

    root = logging.getLogger(__name__)
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch) 

    ##x_test
    logger = logging.getLogger(__name__)
    rs = numpy.random.RandomState(12357)

    ##
    x_test, x_train, y_train, x_val, y_val = read_data()
    idx = dt.create_validation_set(y_train, imbalance=True)

