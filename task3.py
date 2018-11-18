import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy
import pandas
import logging 

from enum import Enum
from sklearn.preprocessing import StandardScaler

import data as dt
import research.fourier as fourier
import research.boosting as boosting
import research.svm as svm 


#######################################################################
class Frequency(Enum):

    HERTZ = 300
    SECONDS = 1/300


#######################################################################
def prepare_data(small_sample=None):

    logger = logging.getLogger(__name__)

    if bool(small_sample):

        ##
        logger.debug('Read small sample data')
        data_dir = os.path.join(dt.data_dir(), dt.Tasks.TASK3.value)
        x_train = pandas.read_csv(os.path.join(data_dir, 'X_train.csv'), header=0, index_col=0, nrows=small_sample)
        x_test = pandas.read_csv(os.path.join(data_dir, 'X_test.csv'), header=0, index_col=0, nrows=small_sample)
        y_train = pandas.read_csv(os.path.join(data_dir, 'y_train.csv'), header=0, index_col=0, nrows=small_sample)

    else:

        ##
        logger.debug('Read data')
        data_dir = os.path.join(dt.data_dir(), dt.Tasks.TASK3.value)
        x_train = pandas.read_csv(os.path.join(data_dir, 'X_train.csv'), header=0, index_col=0)
        x_test = pandas.read_csv(os.path.join(data_dir, 'X_test.csv'), header=0, index_col=0)
        y_train = pandas.read_csv(os.path.join(data_dir, 'y_train.csv'), header=0, index_col=0)

    ##
    logger.debug('Fourier transform data')
    freq_x_train_matrix, pow_x_train = fourier.prepare_frequency_data_set(X=x_train, sample_frequency=Frequency.HERTZ.value, normalize=False, scale=False)
    freq_x_test_matrix, pow_x_test = fourier.prepare_frequency_data_set(X=x_test, sample_frequency=Frequency.HERTZ.value, normalize=False, scale=False)

    assert pow_x_test == pow_x_train

    ##
    freq_x_train = pandas.DataFrame(freq_x_train_matrix)
    freq_x_test = pandas.DataFrame(freq_x_test_matrix)

    return x_train, x_test, y_train, freq_x_test, freq_x_train


#######################################################################
def pre_processing():

    """
    Step 0. Data
    """
    x_train, x_test, y_train, freq_x_test, freq_x_train = prepare_data(small_sample=1000)

    idx_oos_test = dt.create_validation_set(
        y_train=y_train, 
        imbalance=True,
        enforce_imbalance_ratio=False
    )

    orig_freq_x_train = freq_x_train.copy()
    orig_freq_x_test = freq_x_test.copy()
    orig_y_train = y_train.copy()
    logger.debug('dimensions of X_train [{:d}] x [{:d}]'.format(*orig_freq_x_train.shape))
    logger.debug('true distribution of y-values is [{}]'.format(numpy.bincount(orig_y_train.y.values) / len(orig_y_train.y.values)))

    ##
    freq_x_val = freq_x_train.reindex(index=idx_oos_test)
    y_val = y_train.reindex(index=idx_oos_test)

    freq_x_train.drop(idx_oos_test, inplace=True)
    y_train.drop(idx_oos_test, inplace=True)
    logger.debug('dimensions after validaiton removed X_train [{:d}] x [{:d}]'.format(*freq_x_train.shape))

    assert freq_x_train.index.isin(idx_oos_test).any() == False
    assert y_train.index.isin(idx_oos_test).any() == False

    """
    Step 1. Pre-processing
    """
    ##
    if False:
        feature_selector_model = boosting.feature_selection(freq_x_train.values, y_train.y.values.flatten(), forest=True)    
        n,k = feature_selector_model.transform(freq_x_train).shape
        logger.info('number of factors [{:d}] after LinearSVC/RandomForestClassifier model'.format(int(k)))
        freq_x_train = pandas.DataFrame(feature_selector_model.transform(freq_x_train), index=freq_x_train.index)
        freq_x_val = pandas.DataFrame(feature_selector_model.transform(freq_x_val), index=freq_x_val.index)
        freq_x_test = pandas.DataFrame(feature_selector_model.transform(freq_x_test), index=freq_x_test.index)
        check_n, check_k = feature_selector_model.transform(orig_freq_x_test).shape

        assert k == check_k,\
            'error in feature selector model, train and test dimensions do not agree'

    if True:
        bins = 2**8
        density = True
        _,K = freq_x_train.shape 

        ##
        def bound_functor(x):
            return x.reshape(bins, K//bins).mean(-1)

        res = numpy.apply_along_axis(bound_functor, arr=freq_x_train, axis=1)        
        freq_x_train = pandas.DataFrame(res, index=freq_x_train.index)
        del res

        ##
        res = numpy.apply_along_axis(bound_functor, arr=freq_x_val, axis=1)        
        freq_x_val = pandas.DataFrame(res, index=freq_x_val.index)
        del res

        ##
        res = numpy.apply_along_axis(bound_functor, arr=freq_x_test, axis=1)        
        freq_x_test = pandas.DataFrame(res, index=freq_x_test.index)
        del res, bound_functor

        assert freq_x_train.shape[1] == freq_x_test.shape[1]
        assert freq_x_train.shape[1] == freq_x_val.shape[1]

        logger.debug('binning reduced features to [{:d}]'.format(freq_x_train.shape[1]))


    if True:

        """
        mask = freq_x_train.std() > freq_x_train.std().mean() + 0.00 * freq_x_train.std().std()
        keep = mask[mask].index
        logger.debug('variance filter removed [{:d}] features and ended up with [{:d}]'.format(bins-len(keep), len(keep)))
        freq_x_train = freq_x_train.reindex(columns=keep)
        freq_x_test = freq_x_test.reindex(columns=keep)
        freq_x_val = freq_x_val.reindex(columns=keep)
        """
        scale = StandardScaler()
        scale.fit(pandas.concat([freq_x_train, freq_x_test, freq_x_val], axis=0))

        freq_x_train = scale.transform(freq_x_train)
        freq_x_val = scale.transform(freq_x_val)
        freq_x_test = scale.transform(freq_x_test)

    return freq_x_train, freq_x_val, freq_x_test, y_train, y_val


#######################################################################
def svm_path():

    ##
    logger = logging.getLogger(__name__)

    """
    Step 1. Data and pre-processing 
    """
    freq_x_train, freq_x_val, freq_x_test, y_train, y_val = pre_processing()

    """
    Step 2. Fitting (round 1)
    """
    logger.debug('STRICTLY MODEL PARAMETERS - COMMON TO FIRST AND SECOND STAGE')

    c_penalty_lower = 0
    c_penalty_upper = 2
    c_penalty_num = 10
    g_lower = -1
    g_upper = 0
    g_num = 2
    class_weight =  {0:1.68877888, 1:11.55079007,  2:3.47150611, 3:30.1}
    kernel = 'rbf'
    machines = 4

    classifier_kwargs = dict(
    ## data
    x_train=freq_x_train,
    y_train=y_train,
    x_test=freq_x_val,
    y_test=y_val,
    ## params
    c_penalty_lower=c_penalty_lower,
    c_penalty_upper=c_penalty_upper,
    c_penalty_num=c_penalty_num,
    g_lower=g_lower,
    g_upper=g_upper,
    g_num=g_num,
    kernel=kernel,
    class_weight=class_weight,
    ## aux
    machines=machines
    )

    args_to_report = [
        'c_penalty_lower',
        'c_penalty_upper',
        'c_penalty_num',
        'g_lower',
        'g_upper',
        'g_num',
        'kernel',
    ]

    comment_kwargs = {key: classifier_kwargs[key] for key in args_to_report}
    comment = 'r1_{}'.format(comment_kwargs).replace(' ', '').replace("'","")

    logger.info('Running First Stage SVMC, parameters defined by: \n\n [{:s} \n\n]'.format(comment))
    prediction, model = svm.train_svm_classifier(**classifier_kwargs, comment=comment)

    ##
    classes = numpy.unique(prediction)
    counts = numpy.array([(prediction == i).sum() for i in classes])
    ratios = counts / len(prediction)
    logger.info('<y-predict (w/o validation data) \n classes: [{}], \n class ratios [{}], \n class counts [{}]> \n'.format(classes, ratios, counts))
    logger.info('Stage one model diagnostic \n\n')
    print(model)


#######################################################################
def ada_boost_path():

    ##
    logger = logging.getLogger(__name__)

    """
    Step 1. Data and pre-processing 
    """
    freq_x_train, freq_x_val, freq_x_test, y_train, y_val = pre_processing()

    """
    Step 2. Fitting (round 1) 
    """
    
    ##
    logger.debug('STRICTLY MODEL PARAMETERS - COMMON TO FIRST AND SECOND STAGE')
    max_depth = 4
    n_estimators = [100]
    learning_rate_lower = 0 #numpy.log10(0.008858667904100823)
    learning_rate_upper = 0 #numpy.log10(0.008858667904100823)
    learning_rate_num = 1
    machines = 4
    class_weight = 'balanced' #{0:1.706, 1:25, 2:3.582, 3:35} #'balanced'
    criterion = 'gini'
    
    classifier_kwargs = dict(
        ## data
        x_train=freq_x_train, 
        y_train=y_train, 
        x_test=freq_x_val, 
        y_test=y_val,
        ## params
        max_depth=max_depth, 
        n_estimators=n_estimators,
        learning_rate_lower=learning_rate_lower,
        learning_rate_upper=learning_rate_upper,
        learning_rate_num=learning_rate_num,
        class_weight=class_weight,
        criterion=criterion,    
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
    comment = 'r1_{}'.format(comment_kwargs).replace(' ', '').replace("'","")

    logger.info('Running First Stage AdaBoostClassifier, parameters defined by: \n\n [{:s} \n\n]'.format(comment))
    prediction, model = boosting.train_ada_boost_classifier(**classifier_kwargs, comment=comment)

    ##
    classes = numpy.unique(prediction)
    counts = numpy.array([(prediction == i).sum() for i in classes])
    ratios = counts / len(prediction)
    logger.info('<y-predict (w/o validation data) \n classes: [{}], \n class ratios [{}], \n class counts [{}]> \n'.format(classes, ratios, counts))
    logger.info('Stage one model diagnostic \n\n')
    print(model)


#######################################################################
if __name__ == '__main__':

    root = logging.getLogger(__name__)
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch) 

    logger = logging.getLogger(__name__)
    ada_boost_path()