import os
import sys
import pandas
import numpy

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

import regression as dr
import data as dt


"""
Submission text:
Data:
- We used test, train and sample .csv files in order to complete this task
- We did not perform any transformations to the data 
Technique:
- Training and test data was read into python via the pandas module and the read_csv function 
- A simple function for regression was written which takes as arguments 
X := the data as numpy.ndarray / pandas.DataFrame
y := the target/dependent variable as numpy.array / pandas.Series
- The function computes the regression coefficients as B = (X'X)^-1 * X'y
- Obtaining the betas from the aforementioned function, a prediction, based on future data X_test was made 
- The prediction is made as y_hat = X_test * B 
- The submission file was produced and written to file 
"""


#######################################################################
def read_data():

	test = pandas.read_csv(os.path.join(dt.data_dir(), 'task0', 'test.csv'), header=0, index_col=0)
	train = pandas.read_csv(os.path.join(dt.data_dir(), 'task0', 'train.csv'), header=0, index_col=0)

	return test, train


#######################################################################
def main():

	test, train = read_data()

	##
	yCols = ['y']
	xCols = list(set(train.columns).difference(yCols))

	assert set(yCols).intersection(xCols) == set(),\
		"there is a non-trivial intersection between yCols {} and xCols {}".format(" ".join(yCols), " ".join(xCols))

	##
	betas = pandas.Series(dr.ridge_regression(X=train[xCols], y=train[yCols], lambdaParam=0).flatten(), index=xCols)

	##
	predictY = pandas.Series(test[xCols].dot(betas), index=numpy.arange(10000, 12000))
	predictY.to_csv(os.path.join(dt.output_dir(), 'task0_solution.csv'), index=True, header=yCols, index_label=['Id'])



#######################################################################
if __name__ == '__main__':

	main()