# Support Vector Machines
# The attempt to build a boundary between the two classes with the maximum margin. The c parameter determines the regularization - large c uses points near the boundary (less regulariation, noisier) and small c uses more poitns further from the boundary (more regularization, less noise)

# Get data
from sklearn import datasets
iris = datasets.load_iris()
iris_X_train = iris.data[:-20]
iris_X_test = iris.data[-20:]
iris_y_train = iris.target[:-20]
iris_y_test = iris.target[-20:]

from sklearn import svm

svc = svm.SVC()

# Kernels
# These parameters determine the boundary shape
svc = svm.SVC(kernel='linear') # boundary will be a straight line
svc = svm.SVC(kernel='poly', degree=3) # boundary will be a polynomial of the given degree (quadatric, cubic, etc.). When degree=1, this is the same as a linear kernel
svc = svm.SVC(kernel='rbf') # boundary will be based on a radius aorund each point. Radius size is inversely proportional to parameter 'gamma'