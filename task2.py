import os
import sys
import pandas
import numpy
import logging

import data as dt

#######################################################################
def read_data():

    X_test  = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_test.csv'), header=0, index_col=0)
    X_train = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_train.csv'), header=0, index_col=0)
    y_train = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'y_train.csv'), header=0, index_col=0)

    y_train.y = y_train.y.astype(int)

    X_validate = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_validate.csv'), header=0, index_col=0) 
    y_validate = pandas.read_csv(os.path.join(dt.data_dir(), 'task2', 'y_validate.csv'), header=0, index_col=0) 

    y_validate.y = y_validate.y.astype(int)

    assert y_validate.index.isin(X_validate.index).all(),\
        'validation set variables y, X have differences in index -> aborting'

    return \
    X_test,\
    X_train.reindex(index=X_train.index.difference(X_validate.index)),\
    y_train.reindex(index=y_train.index.difference(y_validate.index)),\
    X_validate,\
    y_validate


#######################################################################
if __name__ == '__main__':

    from sklearn.decomposition import PCA
    from sklearn import mixture
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
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

    ##
    scale = StandardScaler()
    scale.fit(x_train)
    explained = 0.85
    pca = PCA(explained, whiten=True)
    pca_x_train = pca.fit_transform(scale.transform(x_train.values))
    pca_n, pca_f = pca_x_train.shape
    logger.info('number of factors [{:d}] to explain [{:f}] variance '.format(pca_f, explained))    

    champion_model = mixture.GaussianMixture(3, covariance_type='full', random_state=rs)
    champion_model.fit(pca_x_train)
    
    y_train_hat = champion_model.predict(pca_x_train)
    y_test_hat = champion_model.predict(pca.transform(scale.transform(x_test)))

    BMAC = balanced_accuracy_score(y_train_hat, y_train)
    breakpoint()
    plt.show()
