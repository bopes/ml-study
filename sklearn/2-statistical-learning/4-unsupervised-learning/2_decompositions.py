# Decompositions
import numpy as np

# Principal Component Analysis
# PCA selects the successive components that explain the maximum variance in the observations. For instance, if one features can be predictably computed using the other two, it should be ignored. Put another way, PCA finds the directions in which the data is not flat:

# Create a signal with only 2 useful dimensions
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1,x2,x3]
# Apply PCA
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
pca.explained_variance_ # this shows that the first two features are  useful, but not the third
# Update the data
pca.n_components = 2 # this sets the number of considered components
X_reduced = pca.fit_transform(X) # this transformed the data to meet the new number of components
X_reduced.shape # equals (100, 2) because there are 100 observatinos with 2 features

