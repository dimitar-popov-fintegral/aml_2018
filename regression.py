import numpy
from scipy import linalg


#######################################################################
def ridge_regression(X, y, lambdaParam):
    '''
    computes the analytical solution to the ridge regression problem

    Args:
        :param X:
        :param y:
        :param lambdaParam:

    :return:
    '''

    n, m = X.shape
    return linalg.inv(X.T.dot(X) + lambdaParam * numpy.eye(m)).dot(X.T.dot(y))