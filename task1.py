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
from research.linear_regression import clean_data, CLEAN_MODE
from scipy import stats

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
def regression(seed, start, end, step, cv=3, comment=''):

	logger = logging.getLogger(__name__)

	##
	logger.info('read provided data')
	X_test, X_train, y_train = read_data()
	std_train, std_test,  = transform_data(X_train=X_train, X_test=X_test)

	##
	removed = 0
	for col in std_train.columns:
		data = std_train[col].copy()
		mask = numpy.abs(data) > data.mean() + 3.5 * data.std()
		std_train.loc[mask, col] = numpy.NaN
		removed += sum(mask)
		del data, mask 
	logger.info('removed a total of [{}] elements'.format(removed))

	##
	if True:
		logger.info('fill NaN with 0 i.e. the mean of the standardized random variables')
		std_train.fillna(1e-3, inplace=True)
		std_test.fillna(1e-3, inplace=True)

	elif False:
		logger.info('fill NaN with linear regression model of X_i = f(y)')

		std_train = clean_data(
			predictors=std_train_temp, 
			response=y_train, 
			clean_mode=CLEAN_MODE.RESPONSE
		)

		std_test.fillna(0.0, inplace=True)
		std_test = std_test.reindex(columns=choose)
		del choose		

	##
	logger.info('feature engineering')
	base_columns = std_train.copy().columns
	base_train = std_train.copy()
	base_test = std_test.copy()
	
	names =  base_columns + '_sq'
	train_sq = base_train.pow(2)
	train_sq.columns = names
	std_train = pandas.concat([std_train, train_sq], axis=1)

	test_sq = base_test.pow(2)
	test_sq.columns = names
	std_test = pandas.concat([std_test, test_sq], axis=1)

	names =  base_columns + '_sin'
	train_sq = numpy.sin(base_train)
	train_sq.columns = names
	std_train = pandas.concat([std_train, train_sq], axis=1)

	test_sq = numpy.sin(base_test)
	test_sq.columns = names
	std_test = pandas.concat([std_test, test_sq], axis=1)

	##
	logger.info('use lasso regression with custom set of lambda parameters')
	alphas = seed ** numpy.arange(start, end, step)
	logger.info('alpha parameters := {}'.format(str(["{0:0.2f}".format(i) for i in alphas]).replace("'", "")))
	reg = LassoCV(alphas=alphas, cv=cv, n_jobs=2, random_state=12357)
	model_cv = reg.fit(std_train.values, y_train.values.flatten())
	logger.info('alpha := {:f}'.format(float(model_cv.alpha_)))
	pred = model_cv.predict(std_test)
	resid = y_train.values.flatten() - model_cv.predict(std_train)

	##
	logger.info('plotting of first stage results')
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(17,10))
	f.suptitle('first stage')
	ax1.plot(resid, 'bo')
	tau = numpy.mean(resid) + 1.64 * numpy.std(resid)
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
	reg = LassoCV(alphas=alphas, cv=cv, n_jobs=2, random_state=12357)
	model_cv = reg.fit(std_train_.values, y_train_.values.flatten()) 
	logger.info('alpha := {:f}'.format(float(model_cv.alpha_)))
	pred = model_cv.predict(std_test)
	resid = y_train_.values.flatten() - model_cv.predict(std_train_)

	##
	logger.info('plotting of second stage results')
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(17,10))
	f.suptitle('second stage')
	ax1.plot(resid, 'bo')
	tau = numpy.mean(resid) + 1.6 * numpy.std(resid)
	mask = numpy.abs(resid) > tau
	ax1.plot([i if numpy.abs(i) > tau else None for i in resid], 'ro')
	ax1.set_title('Residuals')
	ax2.scatter(model_cv.predict(std_train), y_train)
	x0,x1 = ax2.get_xlim()
	y0,y1 = ax2.get_ylim()
	ax2.set_aspect((x1-x0)/(y1-y0))
	ax2.set_title('Fitted vs. Actual, RMSE := {:.6f}'.format(mean_squared_error(y_train, model_cv.predict(std_train))))

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
	cv = 5
	seed=0.2
	start=0.55
	end=1
	step=0.01
	regression(comment='seed_{}_-_start_{}_-_end_{}_-_step_{}_-_cv_{}'.format(seed, start, end, step, cv), seed=seed, start=start, end=end, step=step, cv=cv)
	# classification('grid_search_CV')
	plt.show()

	logger.info('done')