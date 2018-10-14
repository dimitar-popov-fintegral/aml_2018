import os
import sys
import pandas
import numpy
import logging

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

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


# def em_gmm_vect(xs, pis, mus, sigmas, tol=0.01, max_iter=100):

#     n, p = xs.shape
#     k = len(pis)

#     ll_old = 0
#     for i in range(max_iter):
#         exp_A = []
#         exp_B = []
#         ll_new = 0

#         # E-step
#         ws = np.zeros((k, n))
#         for j in range(k):
#             ws[j, :] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs)
#         ws /= ws.sum(0)

#         # M-step
#         pis = ws.sum(axis=1)
#         pis /= n

#         mus = np.dot(ws, xs)
#         mus /= ws.sum(1)[:, None]

#         sigmas = np.zeros((k, p, p))
#         for j in range(k):
#             ys = xs - mus[j, :]
#             sigmas[j] = (ws[j,:,None,None] * mm(ys[:,:,None], ys[:,None,:])).sum(axis=0)
#         sigmas /= ws.sum(axis=1)[:,None,None]

#         # update complete log likelihoood
#         ll_new = 0
#         for pi, mu, sigma in zip(pis, mus, sigmas):
#             ll_new += pi*mvn(mu, sigma).pdf(xs)
#         ll_new = np.log(ll_new).sum()

#         if np.abs(ll_new - ll_old) < tol:
#             break
#         ll_old = ll_new

#     return ll_new, pis, mus, sigmas


#######################################################################
def gmm_classifier(x_train, pi, mu, sigma, tol=1e-7, max_iter=100):

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
	from scipy.stats import multivariate_normal as mvn
	logger = logging.getLogger(__name__)

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
		logger.info('<-- E-Step: construct the expectation -->')
		'''
		Formulation: Calculate the responsibilities g_j(x_i) = P( z_i == j| x_i )
			- formally g_j(x_i) = A / B 
				where A = P( x_i | z_i == j ) * P( z_i == j )
					  B = SUM_{j<=k} P( x_i | z_i == j ) * P( z_i == j )  
			- create the expectation function E_{Z|X} ln P( x_i, z_i == j )
				where E_{Z|X} ln P( x_i, z_i == j ) = SUM P( z_i == j | x_i ) * ln P( x_i, z_i == j )
		'''

		responsibilities = numpy.zeros((k,n))
		for j in range(k):
			responsibilities[j,:] = pi[ij] * mvn(mu[j], sigma[j]).pdf(x_train)
		
		normalization = numpy.ones(k).T.dot(responsibilities).dot(numpy.ones(n))
		responsibilities /= normalization

		logger.info('<-- M-Step: maximize expectation w.r.t parameters T -->')
		'''
		Formulation: Find the ArgMax. w.r.t model parameters, using calculated responsibilities

		By taking derivative w.r.t mu, sigma, pi of the expectation we found in E-step
		we can find the new paramters as follows

			- formally mu_j = SUM_{i<=n} g_j(x_i) * x_i
					   sigma_j = SUM_{i<=n} g_j(x_i) (x_i - mu_j)'(x_i - mu_j) / SUM_{i<=n} g_j(x_i)
					   pi_j = (1/n) * SUM_{i<=n} g_j(x_i) 
		'''		



		P = numpy.zeros((k,n))
		for j in range(k):
			P[j, :] = pi[j] * mvn(mu[j], sigma[j]).pdf(x_train)

		normalization = P.sum(0)
		P /= normalization
		del normalization

		logger.info('<-- M-Step: maximize expectation w.r.t parameters T -->')
		'''
		Formulation
			TODO
		'''
		pi = P.sum(axis=1) / n

		mu = numpy.dot(P, x_train)
		normalization = P.sum(1)[:, None]
		mu /= normalization
		del normalization

		sigma = numpy.zeros((k,f,f))
		for j in range(k):
			deviation = x_train - mu[j, :]
			sigma[j] = (P[j,:,None,None] * numpy.matmul(deviation.values[:,:,None], deviation.values[:,None,:])).sum(axis=0)
		normalization = P.sum(axis=1)[:,None,None]
		sigma /= normalization
		del normalization

		ll_1 = 0.0
		for p, m, s in zip(pi, mu, sigma):
			ll_1 += p * mvn(m, s).pdf(x_train)
		ll_1 = numpy.log(ll_1).sum()

		logger.info('<-- Step [{}] - log-likelihood n [{:f}] -> log-likelihood n+1 [{:f}] -->'.format(step, ll_0, ll_1))

		if numpy.abs(ll_0 - ll_1) < tol:
			break
		ll_0 = ll_1

	return mu, sigma, pi


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
	logger = logging.getLogger(__name__)
	x_test, x_train, y_train = read_data()
	
	x_train = (x_train - x_train.mean()) / x_train.std()
	x_train = x_train.iloc[:, 175:177]
	k = len(y_train.y.unique())
	n, j = x_train.shape

	pi = numpy.random.random(k)
	pi /= pi.sum()

	mu = numpy.zeros((k, j))
	sigma = numpy.array([numpy.eye(j)] * k)

	mu, sigma, pi = gmm_classifier(
		x_train=x_train, 
		pi=pi, 
		mu=mu, 
		sigma=sigma
	)
	import pdb;pdb.set_trace()
