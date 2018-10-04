import numpy
from scipy import linalg


#######################################################################
def ridge_regression(X, y, lambdaParam):
    '''
    Computes the analytical solution to the ridge regression problem

    Args:
        :param X: (numpy.ndarray) features in NxK matrix 
        :param y: (numpy.array) target/dependent variable Nx1
        :param lambdaParam: (float) penalty parameter >= 0 

    '''

    n, m = X.shape
    return linalg.inv(X.T.dot(X) + lambdaParam * numpy.eye(m)).dot(X.T.dot(y))