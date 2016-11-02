# Exercise 2
# Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features. Leave out 10% of each class and test prediction performance on these observations.
# Warning: the classes are ordered, do not leave out the last 10%, you would be testing on only one class.
# Hint: You can use the decision_function method on a grid to get intuitions.

# My Solution
# Get data
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Reduce to first two features
X = X[y != 0, :2]
y = y[y != 0]
# Randomize data
import numpy as np
np.random.seed(0)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
# Prep data
length = len(X)
X_train = X[:.9 * length]
y_train = y[:.9 * length]
X_test = X[.9 * length:]
y_test = y[.9 * length:]
# SVM
from sklearn import svm
for kernel in ['linear','poly','rbf']:
  svc = svm.SVC(kernel=kernel, gamma=10)
  svc.fit(X_train, y_train)
  print("%s: %f" % (kernel, svc.score(X_test, y_test)))


# Solution
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y != 0, :2] # Reduce to two features
y = y[y != 0]
n_sample = len(X)
np.random.seed(0) # Randomize data
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)
X_train = X[:.9 * n_sample] # Prep data
y_train = y[:.9 * n_sample]
X_test = X[.9 * n_sample:]
y_test = y[.9 * n_sample:]
# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)
    print("%s: %f" % (kernel, clf.score(X_test, y_test)))
    # Create plots to show results
    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)
    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    plt.title(kernel)
plt.show()
