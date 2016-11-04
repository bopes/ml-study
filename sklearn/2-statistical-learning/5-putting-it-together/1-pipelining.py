# Pipelining
# Pipe lining lets you create combined estimators

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Create estimators
logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
# Create pipeline
pipe = Pipeline(steps=[('pca', pca),('logistic', logistic)])
# Prep data
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
# fit PCA
pca.fit(X_digits)
# Plot PCA data
# import matplotlib.pyplot as plt
# plt.figure(1,figsize=(4,3))
# plt.clf()
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_')
# plt.axes([.2,.2,.7,.7])
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.show()

# Prediction
import numpy as np
n_components = [20,40,64]
Cs = np.logspace(-4,4,3)

estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs)) # pipeline  paramters can be set with '__' separated paramter names
# This estimator performs everythin included in the pipeline
estimator.fit(X_digits, y_digits)