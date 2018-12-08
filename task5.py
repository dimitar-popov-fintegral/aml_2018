import os
import sys
import logging
import numpy
import pandas
import pickle

from enum import Enum
from collections import namedtuple

import data as dt


#######################################################################
class Frequency(Enum):

    HERTZ = 128
    SECONDS = 4
    SUBJECTS = 3


#######################################################################
def reshape_matrix(X):
    """
    Desc:
        reshapes the input matrices for the task into new dimensions 
        rx = Matrix[SUBJECTS x EPOCHS x SAMPLES[Hz]]
    """
    logger = logging.getLogger(__name__)

    ##
    logger.info('original shape [{shape}]'.format(shape=X.shape))
    seconds_per_day = 24*(60*60)
    epochs_per_day = int(seconds_per_day // Frequency.SECONDS.value)
    rX = X.reshape(Frequency.SUBJECTS.value, epochs_per_day, Frequency.SECONDS.value * Frequency.HERTZ.value)
    logger.info('reshaped [{shape}]'.format(shape=rX.shape))

    ##
    check = rX[0,0,:].flatten()
    numpy.testing.assert_array_equal(X[0,:], check)
    del check 

    ##
    check = rX[-1,-1,:].flatten()
    numpy.testing.assert_array_equal(X[-1,:], check)
    del check

    ##
    check = rX[0, epochs_per_day-1, :].flatten()
    numpy.testing.assert_array_equal(X[epochs_per_day-1,:], check)
    
    return rX


#######################################################################
train_files = [    
    'train_eeg1.csv',
    'train_eeg2.csv',
    'train_emg.csv',
    'train_labels.csv',
]
train_names = [i.split('.')[0] for i in train_files]
Train = namedtuple('Train', ' '.join(train_names))


#######################################################################
test_files = [
    'test_eeg1.csv',
    'test_eeg2.csv',
    'test_emg.csv',
]
test_names = [i.split('.')[0] for i in test_files]
Test = namedtuple('Test', ' '.join(test_names))


#######################################################################
def dump_to_binary(X, name):

    ##
    logger = logging.getLogger(__name__)

    full_file_path = os.path.join(dt.data_dir(), 'task5', '{name}.p'.format(name=name))
    with open(full_file_path, 'w+b') as f:
        logger.info('writing X to binary file [{file}]'.format(file=full_file_path))
        pickle.dump(X, f)

    return full_file_path


#######################################################################
def read_data(files, names, container):

    read_array = numpy.array(files)[None,:]
    ##
    logger = logging.getLogger(__name__)

    def read(file):
        logger.info('reading [{file}]'.format(file=file))
        full_file_path = os.path.join(dt.data_dir(), 'task5', '{}'.format(file[0]))
        df = pandas.read_csv(full_file_path, index_col=0, header=0)
        logger.info('finished reading [{file}] of shape [{shape}]'.format(file=file, shape=df.shape))

        return df.values
        
    tensor = numpy.apply_along_axis(read, 0, read_array)
    logger.info('tensor size [{}]'.format(tensor.shape))

    result = numpy.swapaxes(tensor.T, 1, 2)
    logger.info('result tensor size [{}]'.format(result.shape))

    return container(**dict(zip(names, result)))        


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

    ##
    train = read_data(train_files, train_names, Train)
    binary_dump_path = dump_to_binary(train, 'Train')

    ##
    test = read_data(test_files, test_names, Test)
    binary_dump_path = dump_to_binary(test, 'Test')
