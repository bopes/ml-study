# Scores
# Estimators' score method judges quality of fit. Bigger (~1) is better
from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(kernel='linear', C=1)
svc.fit(X_digits[:-100], y_digits[:-100])
svc.score(X_digits[-100:], y_digits[-100:]) # is 0.98 (very good)

# To better measure prediction accuracy, split the data into folds:
import numpy as np

X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)

scores = list()
for k in range(3):
  # Iterate through X folds - pull out one fold for testing, use remaining two for training
  X_train = list(X_folds)
  X_test = X_train.pop(k)
  X_train = np.concatenate(X_train)
  # Iterate through y folds - pull out one fold for testing, use remaining two for training
  y_train = list(y_folds)
  y_test = y_train.pop(k)
  y_train = np.concatenate(y_train)
  # Train for this iteration
  svc.fit(X_train, y_train)
  # Add this iteration to the scores list
  scores.append(svc.score(X_test, y_test))
# Look at the scores list
print(scores)

# The above foldign process is called "KFold cross-validation"