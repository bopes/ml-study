# Linear regression
# Regression is good for continuous distribution (like the risk for diabetes given certain other health criteria)
from __future__ import print_function
import numpy as np
from sklearn import datasets
# Prep data
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
# Create regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
# View linear regression coefficients
regr.coef_ # view coefficients
# View prediction and mean square error
pred = regr.predict(diabetes_X_test)
actual = diabetes_y_test
np.mean((pred - actual)**2) # view error
# View score (0-1) representing relationship between prediction and actual classifications
regr.score(diabetes_X_test, diabetes_y_test) # view score


# Shrinkage
# There is a tradeoff between variance and bias
import matplotlib.pyplot as plt
X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
# This linear regression shows a lot of variance but no bias
plt.figure()
regr = linear_model.LinearRegression()
np.random.seed(0)
for _ in range(6):
  this_X = .1*np.random.normal(size=(2, 1)) + X
  regr.fit(this_X, y)
  plt.plot(test, regr.predict(test))
  plt.scatter(this_X, y, s=3)
# plt.show()
# This Ridge regression shows little variance but has more bias
plt.figure()
regr = linear_model.Ridge(alpha=1)
np.random.seed(0)
for _ in range(6):
  this_X = .1*np.random.normal(size=(2,1)) + X
  regr.fit(this_X, y)
  plt.plot(test, regr.predict(test))
  plt.scatter(this_X, y, s=3)
# plt.show()
# Can experiment with mulitple alphas to minimize left out error
alphas = np.logspace(-4, -1, 6)
[regr.set_params(alpha=alpha)
     .fit(diabetes_X_train, diabetes_y_train,)
     .score(diabetes_X_test, diabetes_y_test)
  for alpha in alphas] # View scores for different values of alpha

# Sparcity
# Some features have little impact on the result. It is good to reduce the coefficients on these features to mitigate the curse of dimensionality
# Lasso model will reduce the values of these coefs to 0, if required. Is a more aggressive approach than the above Ridge model
regr = linear_model.Lasso()
alphas = np.logspace(-4, -1, 6)
scores = [regr.set_params(alpha=alpha)
              .fit(diabetes_X_train, diabetes_y_train)
              .score(diabetes_X_test, diabetes_y_test)
          for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)
regr.coef_ # The final coefficients

# Classification
# Linear regression if a poor choice for classification (like Iris) because it overweights data that's distant from the decision boundary. A better linear approach in these cases is the logistic function (also called sigmoid)
# y = sigmoid(XB - offset) + e = 1 / ( 1 + e^(-XB + offset) ) + e
iris = datasets.load_iris()
iris_X_train = iris.data[:-20]
iris_X_test = iris.data[-20:]
iris_y_train = iris.target[:-20]
iris_y_test = iris.target[-20:]

logistic = linear_model.LogisticRegression(C=1e5)
# the c parameter controls amount of regularization - large c => less regularization. penalty='12' gives shrinkage coefs, penalty='11' gives sparcity
logistic.fit(iris_X_train, iris_y_train)

