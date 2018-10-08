import os
import sys
import pandas
import numpy
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import regression as dr
import data as dt

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))


#######################################################################
def read_data():

	X_test  = pandas.read_csv(os.path.join(dt.data_dir(), 'task1', 'X_test.csv'), header=0, index_col=0)
	X_train = pandas.read_csv(os.path.join(dt.data_dir(), 'task1', 'X_train.csv'), header=0, index_col=0)
	y_train = pandas.read_csv(os.path.join(dt.data_dir(), 'task1', 'y_train.csv'), header=0, index_col=0)

	return X_test, X_train, y_train


#######################################################################
def transform_data(X_train, X_test):

	logger = logging.getLogger(__name__)

	##
	scale_mean = X_train.mean()
	scale_std = X_train.std()

	##
	logger.info('standardize using mean and std. dev. of observed samples')
	std_X_train = (X_train - scale_mean) / scale_std
	std_X_test = (X_test - scale_mean) / scale_std

	return std_X_train, std_X_test

#######################################################################
def gmm(X, model, n_components, prior, covariance_type):

	from sklearn import mixture

	n, m = X.shape

	if model == 'GMM':
		mm = mixture.GaussianMixture(
			n_components=n_components,
			covariance_type=covariance_type,
			max_iter=100
		).fit(X)
	elif model == 'dirichlet_process_GMM':
		if covariance_type=='diag' or covariance_type == 'spherical':
			cov_prior = 1e0
		else:
			cov_prior * numpy.eye(2)

		mm = mixture.BayesianGaussianMixture(
			n_components=n_components,
			covariance_type=covariance_type,
			weight_concentration_prior=numpy.power(10., prior),
			weight_concentration_prior_type='dirichlet_process',
			max_iter=100,
			random_state=12357).fit(X)

	else:
		raise IOError('invalid model type input')

	covariance = numpy.zeros((n_components, m, m))
	for i in range(n_components):
		if covariance_type == 'full':
			covariances[i] = mm.covariances_[i]
		elif covariance_type == 'tied':
			covariances[i] = mm.covariances_
		elif covariance_type == 'diag':
			covariances[i] = numpy.diag(mm.covariances_[i])
		elif covariance_type == 'spherical':
			covariances[i] = numpy.eye(mm.means_.shape[1]) * mm.covariances_[i]

	return X, mm.predict(X), mm.means_, covariances, title


#######################################################################
def classification(comment=''):

	logger = logging.getLogger(__name__)

	##
	logger.info('read provided data')
	X_test, X_train, y_train = read_data()
	scale_mean = X_train.mean()
	scale_std = X_train.std()

	##
	logger.info('standardize using mean and std. dev. of observed samples')
	std_X_train = (X_train - scale_mean) / scale_std
	std_X_test = (X_test - scale_mean) / scale_std
	y_train.index = y_train.index.astype(int)
	y_train.y = y_train.y.astype(int)


	## 
	logger.info('basic pre-processing of class labels')
	check_obs_per_age = {i: sum(y_train.y == i) for i in y_train.y.unique()}
	remove_low_obs_per_age = {i: j for i, j in check_obs_per_age.items() if j < 7}
	mask = ~y_train.y.isin(remove_low_obs_per_age.keys())

	y_train = y_train[mask]
	std_X_train = std_X_train[mask]
	del mask

	##
	logger.info('fill NaN with 0 i.e. the mean of the standardized random variables')
	std_X_train.fillna(0, inplace=True)
	std_X_test.fillna(0, inplace=True)


	##
	logger.info('grid search for optimum parameter \'alpha\'') 
	mlpclass = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3), alpha=1e-5, random_state=12357, max_iter=1e3)
	paramters = {
		'hidden_layer_sizes': numpy.arange(3, 5, 1),
		'alpha': 10.0 ** -numpy.arange(1, 5, 1.0)
	}
	optMPL = GridSearchCV(mlpclass, paramters)
	optMPL.fit(std_X_train, y_train.values.flatten())

	##
	logger.info('prediction in-sample')
	opt_y_predict = optMPL.predict(std_X_train)

	##
	logger.info('produce classification statistics')
	dim = len(opt_y_predict)
	inSampleConfusionMatrix = confusion_matrix(y_train, opt_y_predict)
	accuracy = numpy.sum(numpy.diag(inSampleConfusionMatrix)) / dim
	logger.info("Using optimal parameter alpha, model accuracy {:.4f} ".format(accuracy))

	##
	logger.info('final prediction')
	opt_y_hat_predict = optMPL.predict(std_X_test)

	##
	logger.info('write to pandas Series object')
	write_to_file = pandas.Series(opt_y_hat_predict, index=X_test.index.astype(int), name='y')
	write_to_file.to_csv(os.path.join(dt.output_dir(), 'task1_solution_{}.csv'.format(comment)), index=True, header=['y'], index_label=['id'])


#######################################################################
def regression(seed, start, end, step, cv=3, comment=''):

	logger = logging.getLogger(__name__)

	##
	logger.info('read provided data')
	X_test, X_train, y_train = read_data()
	std_train, std_test,  = transform_data(X_train=X_train, X_test=X_test)

	# ##
	# logger.info('basic pre-processing of class labels')
	# check_obs_per_age = {i: sum(y_train.y == i) for i in y_train.y.unique()}
	# remove_low_obs_per_age = {i: j for i, j in check_obs_per_age.items() if j < 7}
	# mask = ~y_train.y.isin(remove_low_obs_per_age.keys())

	# y_train = y_train[mask]
	# std_X_train = std_X_train[mask]
	# del mask

	##
	logger.info('fill NaN with 0 i.e. the mean of the standardized random variables')
	std_train.fillna(0.0, inplace=True)
	std_test.fillna(0.0, inplace=True)

	##
	logger.info('use lasso regression with custom set of lambda parameters')
	alphas = seed ** numpy.arange(start, end, step)
	logger.info('alpha parameters := {}'.format(str(["{0:0.2f}".format(i) for i in alphas]).replace("'", "")))
	reg = LassoCV(alphas=alphas, cv=cv)
	model_cv = reg.fit(std_train.values, y_train.values.flatten())
	logger.info('alpha := {:f}'.format(float(model_cv.alpha_)))
	pred = model_cv.predict(std_test)
	resid = y_train.values.flatten() - model_cv.predict(std_train)

	##
	logger.info('plotting of preliminary results')
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(17,10))
	ax1.plot(resid, 'bo')
	tau = numpy.mean(resid) + 1.6 * numpy.std(resid)
	mask = numpy.abs(resid) > tau
	ax1.plot([i if numpy.abs(i) > tau else None for i in resid], 'ro')
	ax1.set_title('Residuals')
	ax2.scatter(model_cv.predict(std_train), y_train)
	x0,x1 = ax2.get_xlim()
	y0,y1 = ax2.get_ylim()
	ax2.set_aspect((x1-x0)/(y1-y0))
	ax2.set_title('Fitted vs. Actual')

	##
	logger.info('use second lasso regression, removing large error inducing observations')
	std_train_ = std_train[~mask]
	y_train_ = y_train[~mask]
	model_cv = reg.fit(std_train_.values, y_train_.values.flatten()) 
	logger.info('alpha := {:f}'.format(float(model_cv.alpha_)))
	pred = model_cv.predict(std_test)
	resid = y_train.values.flatten() - model_cv.predict(std_train)

	##
	logger.info('plotting of final results')
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(17,10))
	ax1.plot(resid, 'bo')
	tau = numpy.mean(resid) + 2 * numpy.std(resid)
	mask = numpy.abs(resid) > tau
	ax1.plot([i if numpy.abs(i) > tau else None for i in resid], 'ro')
	ax1.set_title('Residuals')
	ax2.scatter(model_cv.predict(std_train), y_train)
	x0,x1 = ax2.get_xlim()
	y0,y1 = ax2.get_ylim()
	ax2.set_aspect((x1-x0)/(y1-y0))
	ax2.set_title('Fitted vs. Actual')

	##
	logger.info('write to pandas Series object')
	write_to_file = pandas.Series(pred, index=X_test.index.astype(int), name='y')
	write_to_file.to_csv(os.path.join(dt.output_dir(), 'task1_solution_{}.csv'.format(comment)), index=True, header=['y'], index_label=['id'])


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
	cv = 10
	seed=0.2
	start=0
	end=1
	step=0.01
	regression(comment='seed_{}_-_start_{}_-_end_{}_-_step_{}_-_cv_{}'.format(seed, start, end, step, cv), seed=seed, start=start, end=end, step=step, cv=cv)
	# classification('grid_search_CV')
	plt.show()

	logger.info('done')