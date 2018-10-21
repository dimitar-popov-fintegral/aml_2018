import numpy
import logging
import os 
import sys
import copy 

import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


#######################################################################
def generative_model(x_train, num_components, samples, explained):
    
    logger = logging.getLogger(__name__)
    rs = numpy.random.RandomState(12357)    

    ##
    scale = StandardScaler()
    scale.fit(x_train)

    pca = PCA(explained, whiten=True)
    pca_x_train = pca.fit_transform(scale.transform(x_train.values))
    pca_n, pca_f = pca_x_train.shape
    logger.info('number of factors [{:d}] to explain [{:f}] variance '.format(pca_f, explained))    

    genr_model = mixture.GaussianMixture(3, covariance_type='full', random_state=rs)
    genr_model.fit(pca_x_train)
    
    y_train_hat = genr_model.predict(pca_x_train)
    logger.info('genr. model predictive balanced accuracy score [{:f}]'.format(balanced_accuracy_score(y_train_hat, y_train)))

    genr_data = gmm.sample(samples, random_state=rs)
    genr_samples = pca.inverse_transform(data_new)

    return genr_samples


#######################################################################
def gmm_classifier(x_train, pi, mu, sigma, tol=1e-1, max_iter=100):

    """
    Solving the expectation-maximization (EM) problem for a mixture of Gaussians  
    
    Formulation: 
        - Assume that data are drawn from mixture distribution pi_k * f_k(x; mu_k, sigma_k)
        - Assume that latent variable z determines membership into one of k-classes
        - In the E-step we compute the expectation over all classes, for each data point x_i 
        - In the M-step we max the result of the E-step w.r.t the parameters (T) i.e. mu, sigma

    E-step:
        Determine function E_(Z|X,T_n) [ log { P( X, z | T ) } ]

    M-step:
        argmax_T E_(Z|X,T_n) [ log { P( X, z | T ) } ]
    """

    logger = logging.getLogger(__name__)
    pi_init = copy.deepcopy(pi)
    mu_init = copy.deepcopy(mu)
    sigma_init = copy.deepcopy(sigma)

    ##
    n, f = x_train.shape
    k = len(pi)

    assert len(sigma) == k,\
        'provided sigma dim. [{}] does not conform to supplied mu dim. [{}]'.format(len(sigma), len(mu))
    assert len(pi) == k,\
        'provided number of classes does not match supplied data'

    ll_0 = 0
    for step in range(max_iter):

        ##
        logger.debug('<--STEP{}-->'.format(step))
        logger.info('<-- E-Step: construct the expectation -->')
        '''
        Formulation: Calculate the responsibilities g_j(x_i) = P( z_i == j| x_i )
            - formally g_j(x_i) = A / B 
                where A = P( x_i | z_i == j ) * P( z_i == j )
                      B = SUM_{j<=k} P( x_i | z_i == j ) * P( z_i == j )  
                - create the expectation function E_{Z|X} ln P( x_i, z_i == j )
                where E_{Z|X} ln P( x_i, z_i == j ) = SUM P( z_i == j | x_i ) * ln P( x_i, z_i == j )
        '''

        # E-step
        rs = numpy.zeros((k, n))
        for j in range(k):
            rs[j, :] = pi[j] * mvn(mu[j], sigma[j]).pdf(x_train)
        rs /= rs.sum(0)

        # M-step
        pis = rs.sum(axis=1)
        pis /= n

        mus = numpy.dot(rs, x_train)
        mus /= rs.sum(1)[:, None]

        sigmas = numpy.zeros((k, f, f))
        for j in range(k):
            ys = x_train - mus[j, :]
            sigmas[j] = (rs[j,:,None,None] * numpy.matmul(ys[:,:,None], ys[:,None,:])).sum(axis=0)
        sigmas /= rs.sum(axis=1)[:,None,None]

        ll_1 = 0.0
        for p, m, s in zip(pis, mus, sigmas):
            ll_1 += p * mvn(m, s).pdf(x_train)
        ll_1 = numpy.log(ll_1).sum()

        logger.info('<-- Step [{}] - log-likelihood n [{:f}] -> log-likelihood n+1 [{:f}] -->'.format(step, ll_0, ll_1))

        if numpy.abs(ll_0 - ll_1) < tol:
            break
        ll_0 = ll_1

    return mus, sigmas, pis, rs


#######################################################################
def gmm_experiment():

    ##
    logger = logging.getLogger(__name__)
    rs = numpy.random.RandomState(12357)

    samples_per_component = 1000

    xmat = numpy.vstack([
        rs.multivariate_normal([50, 50], 100 * numpy.eye(2), samples_per_component), 
        rs.multivariate_normal([25, 25], 10  * numpy.eye(2), samples_per_component),]
    )

    _, num_components = xmat.shape

    ymat = numpy.vstack([
        numpy.zeros((samples_per_component, 1)), 
        numpy.ones((samples_per_component, 1))]
    )

    num_train = 1200
    idx = rs.choice(range(num_train), num_train, replace=False)
    logger.info("introducing class imbalance of [{:f}]".format((num_train-samples_per_component)/(num_components * samples_per_component)))

    ##
    x_train = xmat[idx,:].reshape((num_train, num_components))
    y_train = ymat[idx,:].reshape((num_train, 1))

    mask = (y_train==0).flatten()
    data = (x_train[mask], x_train[~mask])
    colours = ('red', 'blue')
    groups = ('wide variance', 'narrow variance')

    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    fig.suptitle('Training data - Red:=high.std. Blue:=low.std.', fontsize=16)

    for dat, colour, group in zip(data, colours, groups):
        x,y = dat[:,0], dat[:,1]
        ax[0].scatter(x, y, alpha=0.8, c=colour, edgecolors='none', label=group)

    ##
    test_idx = list(set(range(num_components*samples_per_component)).difference(set(idx)))

    assert len(test_idx) == num_components*samples_per_component - num_train,\
        'incorrect test index length [{}]'.format(len(test_idx))

    assert len(set(idx).intersection(set(test_idx))) == 0,\
        'non-empty intersection between idx and x_testx_testtest_idx'

    ##
    x_test = xmat[test_idx,:]
    y_test = ymat[test_idx,:]

    mask = (y_test==0).flatten()
    data = (x_test[mask], x_test[~mask]) 

    for dat, colour, group in zip(data, colours, groups):
        x,y = dat[:,0], dat[:,1]
        ax[1].scatter(x, y, alpha=0.8, c=colour, edgecolors='none', label=group)


    n, j = x_train.shape
    k = 2

    logger.info("initialize EM algorithm with solution form k-means clustering")
    kmeans = KMeans(n_clusters=k, random_state=rs).fit(x_train)
    mu = kmeans.cluster_centers_
    sigma = numpy.array([numpy.eye(j)] * k)
    pi = rs.rand(k)
    pi /= pi.sum()

    mu, sigma, pi, responsibilities = gmm_classifier(
        x_train=x_train, 
        pi=pi, 
        mu=mu, 
        sigma=sigma
    )

    ##
    predicted_insample_classes = numpy.argmax(responsibilities.T,axis=1)
    mask = (predicted_insample_classes == 0).flatten()
    data = (x_train[mask], x_train[~mask])

    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    fig.suptitle('Training data predict - Red:=high.std. Blue:=low.std.', fontsize=16)

    for dat, colour, group in zip(data, colours, groups):
        x,y = dat[:,0], dat[:,1]
        ax[0].scatter(x, y, alpha=0.8, c=colour, edgecolors='none', label=group)

    ##
    from scipy.stats import multivariate_normal as mvn
    nTest, fTest = x_test.shape
    rs_pred = numpy.zeros((k, nTest))
    
    for j in range(k):
        rs_pred[j, :] = pi[j] * mvn(mu[j], sigma[j]).pdf(x_test)
    rs_pred /= rs_pred.sum(0)    
    
    predicted_outsample_classes = numpy.argmax(rs_pred.T,axis=1)
    mask = (predicted_outsample_classes == 0).flatten()
    data = (x_test[mask], x_test[~mask])

    for dat, colour, group in zip(data, colours, groups):
        x,y = dat[:,0], dat[:,1]
        ax[1].scatter(x, y, alpha=0.8, c=colour, edgecolors='none', label=group)

    plt.show()


#######################################################################
if __name__ == '__main__':

    root = logging.getLogger(__name__)
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch) 

    gmm_experiment()