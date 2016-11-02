# Grid search
# SKLearn can select parameters to maximize the cross-validation score of an estimator
# By default the GridSearchCV will assume it receives a regressor and use 3-fold cross validation. If it receives a classifier instead, it will use a stratified 3-fold cross validation.
# Nested cross-validation - the GridSearchCV performs two cross_validation loops in parallel - one by the estimator to determine 'gamma' and one by corss_val_score to measure the classifier's prediction performance.

# Setup
from sklearn import datasets, svm
import numpy as np
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
# New imports
from sklearn.model_selection import GridSearchCV
# Create array of C's to try
Cs = np.logspace(-6,-1,10)
# Create and prep grid search object
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),n_jobs=-1)
# Fit grid search object to data
clf.fit(X_digits[:1000],y_digits[:1000])
# Inferred coefs
clf.best_score_ # best score it found ic cross_val_score
clf.best_estimator_.C # best C parameter it found for the estimator
# Predict on test data
clf.score(X_digits[1000:],y_digits[1000:])

# Cross Validated Estimators
# Cross validating a parameter can be done more efficiently for some estimators. For these, there are estimator variants with built-in cross validation (they have CV appended to their name):
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
# Get CV estimator
from sklearn import linear_model
lasso = linear_model.LassoCV()
# Fit CV-estimator to data
lasso.fit(X_diabetes, y_diabetes)
# Check CV-estimator inferred coefs
lasso.alpha_ # Equals 0.01229...