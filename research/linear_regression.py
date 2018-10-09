import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats.mstats import winsorize
from scipy import stats

import data as dt
from enum import Enum


#######################################################################
class CLEAN_MODE(Enum):
    RESPONSE = 'RESPONSE'
    PREDICTORS = 'PREDICTORS'
    ZERO = 'ZERO'


#######################################################################
def clean_data(predictors, response, clean_mode):

    for p in predictors.columns:
        pr = predictors[p].copy()
        pr[abs(stats.zscore(pr.fillna(pr.mean()))) > 2] = np.nan
        predictors[p] = pr

    std_predictors = (predictors-predictors.mean())/predictors.std()
    fstd_predictors = std_predictors.fillna(0.0)

    data = pd.DataFrame(response)
    for p in predictors.columns:

        std_predictor = std_predictors[p]
        idx = predictors.index
        missing_idx = idx[std_predictor.isnull()]
        good_idx = idx.difference(missing_idx)

        if clean_mode == CLEAN_MODE.ZERO:
            print('filling missing predictor [%s] data with zero' % p)
            filled = std_predictor.fillna(0)
        else:
            if clean_mode == CLEAN_MODE.PREDICTORS:
                print('imputing predictor [%s] data based on other predictors' % p)
                X_ = fstd_predictors.reindex(index=good_idx, columns=fstd_predictors.columns.difference([p]))
                x_fill = fstd_predictors.reindex(index=missing_idx, columns=fstd_predictors.columns.difference([p]))

            elif clean_mode == CLEAN_MODE.RESPONSE:
                print('imputing predictor [%s] data based on response' % p)
                X_ = response.reindex(index=good_idx)
                x_fill = response.reindex(index=missing_idx)
            else:
                raise ValueError("unexpected clean mode")

            y_ = std_predictor.reindex(good_idx)
            lm = linear_model.LinearRegression()
            lm.fit(X_, y_)
            y_fill = pd.Series(name=p, index=missing_idx, data=lm.predict(x_fill))
            filled = pd.concat([y_, y_fill]).reindex(idx)
            filled = (filled-filled.mean())/filled.std()

        assert not filled.isnull().values.any()
        assert all(filled.index == data.index)
        data = data.join(filled)

    return data


#######################################################################
def main():
    
    #######################################################################
    ## READ and CLEAN data
    #######################################################################

    predictors = pd.read_csv(os.path.join(dt.data_dir(), 'task1', 'X_train.csv'), header=0, index_col=0)
    zero_variance = predictors.columns[predictors.describe().transpose()['std']<1e-6]
    predictors = predictors.reindex(columns=predictors.columns.difference(zero_variance))
    test_predictors = pd.read_csv(os.path.join(dt.data_dir(), 'task1', 'X_test.csv'), header=0, index_col=0)

    response = pd.read_csv(os.path.join(dt.data_dir(), 'task1', 'y_train.csv'), header=0, index_col=0)

    train_data = clean_data(predictors, response, clean_mode=CLEAN_MODE.RESPONSE)
    predict_data = clean_data(test_predictors, pd.DataFrame(index=test_predictors.index, columns=['y'], data=0), clean_mode=CLEAN_MODE.ZERO)


    #######################################################################
    ## RUN REGRESSION
    #######################################################################

    in_sample_cnt = 1200
    pca_num_components = None
    importance_threshold = 0.0015
    predict = True

    y = train_data[response.columns].values
    y_ = y[0:in_sample_cnt]
    y_test = y[in_sample_cnt:]

    if importance_threshold is not None:
        model = ExtraTreesClassifier()
        model.fit(train_data[predictors.columns].values, train_data[response.columns].values)
        importance = pd.Series(index=predictors.columns, data=model.feature_importances_)
        features = importance[importance >= importance_threshold].index
    else:
        features = predictors.columns

    print("Using %d features" % len(features))
    X = train_data[features].values
    X_predict = predict_data[features].values
    X_train = X[0:in_sample_cnt]
    X_test = X[in_sample_cnt:]


    if pca_num_components is not None:
        pca = PCA(n_components=min(pca_num_components, len(features)))
        pca.fit(X_train)
        ev = pca.explained_variance_ratio_
        print("Explained variance: %s, (total: %s)" % (ev, sum(ev)))
        X_train = pca.transform(X_train)
        X_predict = pca.transform(X_predict)
        if len(X_test) > 0:
            X_test = pca.transform(X_test)


    # Create linear regression object
    lm = linear_model.LinearRegression(normalize=True)
    # Train the model using the training sets
    lm.fit(X_train, y_)

    if predict:
        comment = "PCA_%s_Importance_%s(%s)_DataCnt_%s" % (pca_num_components, importance_threshold, len(features), in_sample_cnt)

        print('predict and write to pandas Series object')
        y = lm.predict(X_predict)
        write_to_file = pd.Series(y.flatten(), index=predict_data.index, name='y')
        write_to_file.to_csv(os.path.join(dt.output_dir(), 'task1_solution_{}.csv'.format(comment)), index=True, header=['y'], index_label=['id'])


    if len(y_test) > 0:

        # Make predictions using the testing set
        y_fill = lm.predict(X_test)

        # The coefficients
        print('Coefficients: \n', lm.coef_)
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_fill))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(y_test, y_fill))

        mpl.style.use('bmh')
        plt.figure()
        plt.plot(y_fill - y_test, 'o')
        plt.title('Residuals')

        plt.figure()
        plt.scatter(y_fill, y_test, color='black')
        plt.title('Predicted vs. Actual')

        plt.show()

#######################################################################
if __name__ == '__main__':

    main()



