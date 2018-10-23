import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import statsmodels.api as sm

from statsmodels.graphics.regressionplots import plot_leverage_resid2
from scipy.stats import zscore, norm
from random import shuffle

import data as dt
from enum import Enum


#######################################################################
class FILL_MODE(Enum):
    RESPONSE = 'RESPONSE'
    PREDICTORS = 'PREDICTORS'
    MEAN = 'MEAN'


#######################################################################
def clean_data(predictors, response, zscore_threshold, fill_mode):

    idx = predictors.index
    if response is None:
        data = pd.DataFrame(columns=['y'], index=idx)
    else:
        data = pd.DataFrame(response)


    r2 = pd.Series(index=predictors.columns)

    for p in predictors.columns:
        predictor = predictors[p]
        predictor[abs(stats.zscore(predictor.fillna(predictor.mean()))) > zscore_threshold] = np.nan

        missing_idx = idx[predictor.isnull()]
        good_idx = idx.difference(missing_idx)

        if fill_mode == FILL_MODE.MEAN:
            print('filling missing predictor [%s] data with mean' % p)
            predictor = predictor.fillna(predictor.mean())
        else:
            if fill_mode == FILL_MODE.PREDICTORS:
                print('imputing predictor [%s] data based on other predictors' % p)
                X = predictors.drop(p, axis=1)
                X = X.fillna(X.mean())
            elif fill_mode == FILL_MODE.RESPONSE:
                print('imputing predictor [%s] data based on response' % p)
                assert response is not None
                X = response
            else:
                raise ValueError("unexpected clean mode")

            X = (X-X.mean()/X.std())
            lm = sm.OLS(predictor.reindex(good_idx).values,
                        sm.add_constant(X.reindex(good_idx).values)).fit()
            r2[p] = lm.rsquared
            r2.name = 'r2'

            missing = pd.Series(name=p, index=missing_idx,
                                data=lm.predict(sm.add_constant(X.reindex(missing_idx).values)))
            predictor = pd.concat([predictor.reindex(good_idx), missing]).reindex(idx)

        predictor = (predictor-predictor.mean())/predictor.std()
        assert not predictor.isnull().values.any()
        assert all(predictor.index == data.index)
        data = data.join(predictor)

    return data, r2


#######################################################################
def diags(response, predictors):
    X = sm.add_constant(predictors.values)
    Y = response.values
    lm = sm.OLS(Y, X).fit()
    return lm.rsquared, lm.aic, pd.Series(index=predictors.columns, data=lm.pvalues[1:])


#######################################################################
def select_aic(candidate_features, y, X):
    test_features = list()
    aic = 0
    R2 = None
    for feature in candidate_features:
        test_features.append(feature)
        new_R2, new_aic, pvals = diags(y, X[test_features])
        if new_aic < aic:
            test_features.remove(feature)
        else:
            aic = new_aic
            R2 = new_R2

    return R2, aic, test_features


#######################################################################
def filter_high_leverage(lm, y, X):

    infl = lm.get_influence()
    leverage = infl.hat_matrix_diag
    resid = zscore(lm.resid)
    alpha = .05
    cutoff = norm.ppf(1.-alpha/2)

    large_resid = np.abs(resid) > cutoff
    lr_idx = pd.Series(large_resid)[large_resid].index
    large_leverage = pd.Series(abs(stats.zscore(leverage))>1.0)
    ll_idx = pd.Series(large_leverage)[large_leverage].index

    idx = X.index.difference(ll_idx.intersection(lr_idx))

    y = y.reindex(idx)
    X = X.reindex(idx)

    return y, X


#######################################################################
def run(SAMPLE_SIZE, PCA_N, PV, AIC, features, train_data):

    y = train_data.loc[0:SAMPLE_SIZE, 'y']
    X = train_data.loc[0:SAMPLE_SIZE, features]

    y_hat = train_data.loc[SAMPLE_SIZE:, 'y']
    X_hat = train_data.loc[SAMPLE_SIZE:, features]

    if AIC is not None:
        candidate_features = list(features)
        R2, aic, features = select_aic(candidate_features, y, X)
        print("Selected %s features, R2[%s], AIC[%s]" % (len(features), R2, aic))
        for i in range(AIC):
            shuffle(candidate_features)
            new_R2, aic, new_features = select_aic(candidate_features, y, X)
            if new_R2 > R2:
                R2 = new_R2
                features = new_features
                print("Selected %s features, R2[%s], AIC[%s]" % (len(features), R2, aic))


    if PV is not None:
        R2, aic, pvals = diags(response, X[features])
        features = pvals[pvals<PV].index

    print("Proceed with %s features" % len(features))
    X = X[features]
    X_hat = X_hat[features]

    lm = sm.OLS(y, sm.add_constant(X)).fit()
    print("Train regression:\n%s" % lm.summary())
    y, X = filter_high_leverage(lm, y, X)
    lm = sm.OLS(y, sm.add_constant(X)).fit()
    print("Re-train regression:\n%s" % lm.summary().tables[0])


    if PCA_N > 0:
        pca = PCA(n_components=min(PCA_N, len(features)))
        pca.fit(X)
        ev = pca.explained_variance_ratio_
        print("Explained variance: %s, (total: %s)" % (ev, sum(ev)))
        X = pca.transform(X)
        if len(X_hat) > 0:
            X_hat = pca.transform(X_hat)

        lm = sm.OLS(y, sm.add_constant(X)).fit()
        print("Regression on PCA:\n%s" % lm.summary().tables[0])
        #y, X = filter_high_leverage(lm, y, X)
        lm = sm.OLS(y, sm.add_constant(X)).fit()
        print("Re-train regression on PCA:\n%s" % lm.summary().tables[0])


    if len(y_hat) > 0:
        y_test_hat = lm.predict(sm.add_constant(X_hat))

        SS_tot=((y_hat - y_hat.mean())**2).sum()
        SS_res=((y_test_hat-y_hat)**2).sum()

        print("R2 (out of sample): %s" % (1-(SS_res/SS_tot)))


        mpl.style.use('bmh')
        plt.figure()
        plt.plot(y_test_hat - y_hat, 'o')
        plt.title('Residuals')

        plt.figure()
        plt.scatter(y_test_hat, y_hat, color='black')
        plt.title('Predicted vs. Actual')

        y_test = lm.predict(sm.add_constant(X))
        plt.figure()
        plt.scatter(y_test, y, color='black')
        plt.title('Predicted vs. Actual (in sample)')

        fig, ax = plt.subplots   (figsize=(8,6))
        fig = plot_leverage_resid2(lm, ax = ax)

        plt.show()



#######################################################################
## DATA CLEAN setup
#######################################################################

T_FILL=FILL_MODE.RESPONSE
P_FILL=FILL_MODE.PREDICTORS
Z = 2.5
RELOAD = False


#######################################################################
## RUN REGRESSION
#######################################################################

if __name__ == "__main__":

    y_ = pd.read_csv(os.path.join(dt.data_dir(), 'task1', 'y_train.csv'), header=0, index_col=0)
    x_ = pd.read_csv(os.path.join(dt.data_dir(), 'task1', 'X_train.csv'), header=0, index_col=0)
    assert all(y_.index == x_.index)

    index_val = dt.create_validation_set(y_)

    y_train = y_.reindex(y_.index.difference(index_val))
    x_train = x_.reindex(x_.index.difference(index_val))
    assert all(y_train.index == x_train.index)

    y_val = y_.reindex(index_val)
    x_val = x_.reindex(index_val)
    assert all(y_val.index == x_val.index)

    zero_variance_cols = dt.small_variance_cols(x_train, threshold=1e-6)
    all_features = x_train.columns.difference(zero_variance_cols)

    dt.remove_outliers(x_train, zscore_threshold=Z)
    print("filled")
    filled = dt.fill_missing_with_ols(x_train, x_train)


    train_data, train_data_r2 = clean_data(all_features, y_train, zscore_threshold=Z, fill_mode=T_FILL)
    train_data.to_csv(os.path.join(dt.output_dir(), ), index=True, header=True)
    train_data_r2.to_csv(os.path.join(dt.output_dir(), train_data_r2_fname), index=True, header=True)

    train_data = pd.read_csv(os.path.join(dt.output_dir(), train_data_fname), header=0, index_col=0)
    train_data_r2 = pd.read_csv(os.path.join(dt.output_dir(), train_data_r2_fname), header=0, index_col=0)['r2']

    features = train_data_r2.sort_values(ascending=False).head(400).index
    SAMPLE_SIZE = 1000
    PCA_N = 120
    PV = None
    AIC = None


    run(SAMPLE_SIZE, PCA_N, PV, AIC, features, train_data)









