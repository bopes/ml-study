# Clustering
# Cluster the data into separate groupings based on their criteria

# The simplest clustering algorithm is K-means clustering:
from sklearn import cluster, datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
# Create classifier
k_means = cluster.KMeans(n_clusters=3)
# Fit classifier
k_means.fit(X_iris)
print(k_means.labels_[::10])
print(y_iris[::10])
# Note - there is no guarantee of recovering the truth ,so don't overinterpret clustering results

# Vector Quantization
# This is a way of using clustering to compress data. A common use for this is posterizing an image, reducing the color scale to a histogram
import scipy as sp
import numpy as np
try:
  face = sp.face(gray=True)
except AttributeError:
  from scipy import misc
  face = misc.face(gray=True)
x = face.reshape((-1,1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(x)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape
# Display images
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 3.6))
plt.imshow(face, cmap=plt.cm.gray) # original face
plt.show()
plt.imshow(face_compressed, cmap=plt.cm.gray) # clustered face
plt.show()


# Hierarchial clustering
# Clusters are based on a hierarchy
  # Agglomerative cluster: each observation has its own cluster, and these are iteratively merged. More computationally efficient for large samples
  # Divisive - all observaions start in one cluster which is split in multiple clusters. Computationally slow for large numbers of clusters

# Connectivirt-constrained clustering
import matplotlib.pyplt as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

if sp_version < (0, 12):
  raise SkipTest("Skipping because SciPy version earlier than 0.12.0 and "
                  "thus does not include the scipy.misc.face() image.")
# Generate data
try:
  face = sp.face(gray=True)
except AttributeError:
  from scipy import misc
  face = misc.face(gray=True)
# Resize to 10% of original to speed up processing
face = sp.misc.imresize(face,0.10)/255.

# Feature agglomeration
# This merges similar features to address sparsity
digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)
agglo = cluster.FeatureAgglomeration(connectivty=connectivty, n_clusters=32)
agglo.fit(X)
X_reduced = agglo.transform(X)
X_approx = agglo.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)