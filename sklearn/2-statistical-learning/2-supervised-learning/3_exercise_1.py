# Exercise 1
# Try classifying the digits dataset with nearest neighbors and a linear model. Leave out the last 10% and test prediction performance on these observations.

# My Attempt
# Import data
from sklearn import datasets
# Prep Data
digits = datasets.load_digits()
samples = len(digits.data)
X_train = digits.data[:.9 * samples]
y_train = digits.target[:.9 * samples]
X_test = digits.data[.9 * samples:]
y_test = digits.target[.9 * samples:]
# Nearest Neighbor prediction
from sklearn import neighbors
nn = neighbors.KNeighborsClassifier()
nn.fit(X_train, y_train)
nn_score = nn.score(X_test, y_test)
# Logistic Regression prediction
from sklearn import linear_model
lr = linear_model.LogisticRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
# Display results
print("My answers:")
print("Nearest Neighbors: %f" % nn_score)
print("Logistic Regression: %f" % lr_score)

# Solution
from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))