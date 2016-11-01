import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

iris_X = iris.data # Raw data
iris_y = iris.target # Known classifications for training

# Split data randomly (seed is used to ensure consistency with tutorial)
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
# Training sets
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
# Testing sets
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create a Nearest Neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# Fit training data
knn.fit(iris_X_train, iris_y_train)

# Make prediction classifications
pred = knn.predict(iris_X_test)
print(pred)
# Compare to actual classifications
print(iris_y_test)

# Curse of Dimensionality
# https://en.wikipedia.org/wiki/Curse_of_dimensionality
# For an estimator to be effective, you need the distance between neighboring points to be less than some value d, which depends on the problem. In one dimension, this requires on average n ~ 1/d points. In the context of the above k-NN example, if the data is described by just one feature with values ranging from 0 to 1 and with n training observations, then new data will be no further away than 1/n. Therefore, the nearest neighbor decision rule will be efficient as soon as 1/n is small compared to the scale of between-class feature variations.
# If the number of features is p, you now require n ~ 1/d^p points. Let’s say that we require 10 points in one dimension: now 10^p points are required in p dimensions to pave the [0, 1] space. As p becomes large, the number of training points required for a good estimator grows exponentially.
# For example, if each point is just a single number (8 bytes), then an effective k-NN estimator in a paltry p~20 dimensions would require more training data than the current estimated size of the entire internet (±1000 Exabytes or so).