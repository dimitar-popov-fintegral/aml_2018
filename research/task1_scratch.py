#######################################################################
def gmm(X, model, n_components, prior, covariance_type):

	from sklearn import mixture

	n, m = X.shape

	if model == 'GMM':
		mm = mixture.GaussianMixture(
			n_components=n_components,
			covariance_type=covariance_type,
			max_iter=100
		).fit(X)mm_classifier
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

	# ##
	# logger.info('basic pre-processing of class labels')
	# check_obs_per_age = {i: sum(y_train.y == i) for i in y_train.y.unique()}
	# remove_low_obs_per_age = {i: j for i, j in check_obs_per_age.items() if j < 7}
	# mask = ~y_train.y.isin(remove_low_obs_per_age.keys())

	# y_train = y_train[mask]
	# std_X_train = std_X_train[mask]
	# del mask

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


