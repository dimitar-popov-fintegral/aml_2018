import os
import numpy as np
import pandas as pd
import data as dt
import statsmodels.api as sm
from scipy import stats


from sklearn.decomposition import PCA

Z = 2
BALANCE=[0.125, 0.75, 0.125]


#######################################################################
def small_variance_cols(x_train, threshold):

    cols_std = x_train.describe().transpose()['std']
    return cols_std[cols_std<threshold].index


#######################################################################
def nan_outliers(x_train, zscore_threshold):

    for p in x_train.columns:
        predictor = x_train[p]
        predictor[abs(stats.zscore(predictor.fillna(predictor.mean()))) > zscore_threshold] = np.nan


#######################################################################
def fill_missing_predictors_with_ols(predictors):

    data = pd.DataFrame(index=predictors.index)

    for p in predictors.columns:
        print(p)

        predictor = predictors[p]

        missing_idx = predictor[predictor.isnull()].index
        good_idx = predictor.index.difference(missing_idx)

        X = predictors.reindex(columns = predictors.columns.difference([p])).fillna(predictors.mean())
        X = X.join(pd.Series(index=predictor.index, data=1.0, name='intercept'))

        Y_train = predictor.reindex(good_idx).values
        X_train = X.reindex(good_idx).values
        X_predict = X.reindex(missing_idx).values
        lm = sm.OLS(Y_train, X_train)
        Y_predict = lm.fit().predict(X_predict)

        missing = pd.Series(name=p, index=missing_idx, data=Y_predict)
        predictor = pd.concat([predictor.reindex(good_idx), missing]).reindex(data.index)

        assert not predictor.isnull().values.any()
        assert all(predictor.index == data.index)
        data = data.join(predictor, how='outer')

    predictors.update(data)


#######################################################################
def standardize(X):
    X.update((X-X.mean())/X.std())


#######################################################################
def read_data(clean=False, num_pca=None, num_el=None):

    y = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'y_train.csv'), header=0, index_col=0)
    X = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_train.csv'), header=0, index_col=0)
    X_test = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_test.csv'), header=0, index_col=0)

    if num_el is not None:
        y = y.iloc[:num_el,:]
        X =  X.iloc[:num_el,:]
        X_test = X_test.iloc[:num_el,:]

    assert all(y.index == X.index)
    assert all(X_test.columns == X.columns)

    if clean:
        X = X.reindex(columns=X.columns.difference(small_variance_cols(X,threshold=1e-6)))
        X_test = X_test.reindex(columns=X.columns)

        nan_outliers(X, zscore_threshold=Z)
        standardize(X)
        fill_missing_predictors_with_ols(X)

        nan_outliers(X_test, zscore_threshold=Z)
        standardize(X_test)
        fill_missing_predictors_with_ols(X_test)

    if num_pca is not None:
        standardize(X)
        standardize(X_test)

        pca = PCA(n_components=num_pca)
        pca.fit(X)
        ev = pca.explained_variance_ratio_
        print("Explained variance: %s, (total: %s)" % (ev, sum(ev)))

        X = pd.DataFrame(data=pca.transform(X))
        X_test = pd.DataFrame(data=pca.transform(X_test))


    assert all(y.index == X.index)
    assert all(X_test.columns == X.columns)

    y.to_csv(os.path.join(dt.output_dir(), 'y_input.csv'))
    X.to_csv(os.path.join(dt.output_dir(), 'X_input.csv'))
    X_test.to_csv(os.path.join(dt.output_dir(), 'X_test_input.csv'))

    return X, y, X_test


#######################################################################
def write_libsvm_input(y, X, file_name):

    assert all(y.index == X.index)

    with open(os.path.join(file_name), 'w') as f:
        for row_n in range(len(y.index)):
            row = "%s " % int(y.iloc[row_n])
            for col_n in range(len(X.columns)):
                row += "%s:%s " % (col_n+1, X.iloc[row_n, col_n])
            row += "\n"
            f.write(row)
