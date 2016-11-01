# Type Casting 1
# Unless otherwise indicated, inputs will be cast to float64
import numpy as np
from sklearn import random_projection
# Generate random array
rng = np.random.RandomState(0)
X = rng.rand(10,2000)
# Cast random array to a float32 np.array
X = np.array(X, dtype='float32')
print(X.dtype) # prints 'float32'
# Transformer recasts X to float64 by default (this happens with all other regression targets as well)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype) # prints 'float64'

# Type Casting 2
# Classification targets are maintained and are not cast to float64
from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
# This classifier returns an array of integers, like iris.target
clf.fit(iris.data, iris.target)
print(list(clf.predict(iris.data[:3])))
# This classifier returns an array of strings because it used iris.target_names for fitting
clf.fit(iris.data, iris.target_names[iris.target])
print(list(clf.predict(iris.data[:3])))


# Refitting and updating parameters
# Classifier parameters can be set manually and overwritten with each new fit
import numpy as np
from sklearn.svm import SVC
# Generate random array
rng = np.random.RandomState(0)
X = rng.rand(100,10)
y = rng.binomial(1,0.5,100)
X_test = rng.rand(5,10)
# Create classifier
clf = SVC()
# Set parameters with the linear kernel and run prediction
clf.set_params(kernel='linear').fit(X,y)
print(clf.predict(X_test)) # prints [1 0 1 1 0]
# Update the parameters and fit to the same dataset and make new prediction
clf.set_params(kernel='rbf').fit(X,y)
print(clf.predict(X_test)) # prints [0 0 0 1 0]