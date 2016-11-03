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

# Independent Component Analysis
# ICA selects components such that the distribution of their loadings carries the maximuum amount of independent info. It is able to reocver non-Gaussian independent signals

# Generate sample data
time = np.linspace(0,10,2000)
s1 = np.sin(2 * time) # Signal 1 - sinusoidal signal
s2 = np.sign(np.sin(3 * time)) # Signal 2 - square signal
S = NP.C_[s1,s2]
S += 0.2 * np.random.normal(size=S.shape) # Add noise
S /= S.std(axis=0) # Standardize data
# Mix data
A = np.array([1,1],[0.5,2]) # Mixing matrix
X = np.dots(S,A.T) # Generate observations
# Compute ICA
ica = decomposition.FastICA()
S_ = ica.fit_transform(X) # get the estimated sources
A_ = ica.mixing_.T
print(np.allclose(X,np.dots(S_,A_) + ica.mean_))