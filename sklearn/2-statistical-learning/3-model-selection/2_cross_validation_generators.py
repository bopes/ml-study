# Cross validation generators
# These contain a split method which allows input to be split and interated through:

from sklearn.model_selection import KFold
# test data
X = ["a","a","b","c","c","c"]
# Create cross validation generator
k_fold = KFold(n_splits=3)
# View how it splits the data (in this case the data is X)
for train_indices, test_indices in k_fold.split(X):
  print("Train: %s | test: %s" % (train_indices,test_indices)) # returns this:
  # Train: [2 3 4 5] | test: [0 1]
  # Train: [0 1 4 5] | test: [2 3]
  # Train: [0 1 2 3] | test: [4 5]

# Real data example
from sklearn import datasets
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

from sklearn import svm
svc = svm.SVC(C=1, kernel='linear')
# Here is a very manual way to do it:
from sklearn.model_selection import KFold
scores = list()
for train, test in k_fold.split(X_digits):
  svc.fit(X_digits[train], y_digits[train])
  score = svc.score(X_digits[test], y_digits[test])
  scores.append(score)
scores

# Here is a slightly simpler way to do it:
from sklearn.model_selection import KFold
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in k_fold.split(X_digits)]

# Here is the simplest way to do it (with a helper):
from sklearn.model_selection import cross_val_score
cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=1) # n_jobs=1 means the computation will be dispatched on all CPUs in the computer

# cross_val_score can use different scoring methods as well:
from sklearn.model_selection import cross_val_score
cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro')

# Available cross-validation generators:
# KFold - splits data into k folds, then trains on k-1 folds and tests on the remaining one
# StratifiedKFold - same as KFold but keeps class distribution within each fold
# GroupKFold - ensures that the same group is not included in both training and testing sets
# ShuffleSplit - generates train/test indices based on random permutation
# StratifiedShuffleSplit - same as ShuffleSplit but keeps class distribution within each iteration
# GroupShuffleSplit - ensures that the same group is not included in both training and testing sets
# LeaveOneOut - leaves one observation out
# LeavePOut - leaves P observations out
# LeaveOneGroupOut - takes a group array to group observations
# LeavePGroupsOut - leaves P groups out
# PredefinedSplit - generates train/test indices based on predefined splits