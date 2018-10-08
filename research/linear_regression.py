import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats.mstats import winsorize

import data as dt

#######################################################################
## READ DATA
#######################################################################

predictors = pd.read_csv(os.path.join(dt.data_dir(), 'task1', 'X_train.csv'), header=0, index_col=0)
response = pd.read_csv(os.path.join(dt.data_dir(), 'task1', 'y_train.csv'), header=0, index_col=0)


#######################################################################
## FILL PREDICTORS DATA
#######################################################################

data = pd.DataFrame(response)
for p in predictors.columns:

    print('imputing predictor: %s' % p)

    idx = response.index
    missing_idx = idx[predictors[p].isnull()]
    good_idx = idx.difference(missing_idx)

    x_train = response.reindex(good_idx)
    y_train = predictors[p].reindex(good_idx)
    x_pred = response.reindex(missing_idx)

    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)
    y_pred = pd.Series(name=p, index=missing_idx, data=lm.predict(x_pred))

    filled = pd.concat([y_train, y_pred]).reindex(idx)


    assert not filled.isnull().values.any()
    assert all(filled.index == data.index)

    data = data.join(filled)



#######################################################################
## RUN REGRESSION
#######################################################################

IN_SAMPLE_CNT = 1000
PCA_NUM_COMPONENTS = None
WINSORIZE = None
IMPORTANCE = 0.0015


y = data[response.columns].values
y_train = y[0:IN_SAMPLE_CNT]
y_test = y[IN_SAMPLE_CNT:]


if IMPORTANCE is not None:
    X = data[predictors.columns].values
    X_train = X[0:IN_SAMPLE_CNT]
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    importance = pd.Series(index=predictors.columns, data=model.feature_importances_)
    features = importance[importance >= IMPORTANCE].index
else:
    features = predictors.columns

print("Using %d features" % len(features))


if PCA_NUM_COMPONENTS is not None:
    pca = PCA(n_components=min(PCA_NUM_COMPONENTS, len(features)))
    X = data[features].values

    # Split the data into training/testing sets
    X_train_ = X[0:IN_SAMPLE_CNT]
    if WINSORIZE is not None:
        X_train_ = winsorize(X_train_, limits=WINSORIZE, axis=0)

    X_test_ = X[IN_SAMPLE_CNT:]

    pca.fit(X_train_)
    ev = pca.explained_variance_ratio_
    print("Explained variance: %s, (total: %s)" % (ev, sum(ev)))

    X_train = pca.transform(X_train_)
    X_test = pca.transform(X_test_)

else:
    X = data[features].values

    # Split the data into training/testing sets
    X_train = X[0:IN_SAMPLE_CNT]
    X_test = X[IN_SAMPLE_CNT:]

    if WINSORIZE is not None:
        X_train = winsorize(X_train, limits=WINSORIZE, axis=0)


# Create linear regression object
lm = linear_model.LinearRegression()

# Train the model using the training sets
lm.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = lm.predict(X_test)

# The coefficients
print('Coefficients: \n', lm.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


#######################################################################
## PLOT RESULTS
#######################################################################

mpl.style.use('bmh')
plt.figure()
plt.plot(y_pred-y_test, 'o')
plt.title('Residuals')

plt.figure()
plt.scatter(y_pred, y_test, color='black')
plt.title('Predicted vs. Actual')

plt.show()



