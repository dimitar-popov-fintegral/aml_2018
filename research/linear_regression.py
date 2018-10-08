import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

import data as dt

OUT_OF_SAMPLE_CNT = 500
NUM_FEATURES = 10
USE_PCA = True

data = pd.read_csv(os.path.join(dt.data_dir(), 'task0', 'train.csv'), header=0, index_col=0)


y = data.iloc[:, 0].values

if USE_PCA:
    pca = PCA(n_components=NUM_FEATURES)
    X_ = data.iloc[:, 1:].values
    pca.fit(X_)
    ev = pca.explained_variance_ratio_
    print("Explained variance: %s, (total: %s)" % (ev, sum(ev)))
    X = pca.transform(X_)
else:
    X = data.iloc[:, -NUM_FEATURES:].values


# Split the data into training/testing sets
X_train = X[:-OUT_OF_SAMPLE_CNT]
X_test = X[-OUT_OF_SAMPLE_CNT:]

# Split the targets into training/testing sets
y_train = y[:-OUT_OF_SAMPLE_CNT]
y_test = y[-OUT_OF_SAMPLE_CNT:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
mpl.style.use('bmh')
plt.figure()
plt.plot(y_pred-y_test, 'o')
plt.title('Residuals')

plt.figure()
plt.scatter(y_pred, y_test, color='black')
plt.title('Predicted vs. Actual')


plt.show()






