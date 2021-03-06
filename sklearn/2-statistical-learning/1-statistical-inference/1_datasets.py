# Datasets
from sklearn import datasets
# Iris data is already prepped for sklearn - it is a 2d array
iris = datasets.load_iris()
data = iris.data
print(data.shape) # (150, 4) - 150 examples with 4 features each

# Digits data is not ready for sklearn - it is a 3d array and needs to be reduced to 2d
digits = datasets.load_digits()
print(digits.images.shape) # (1797, 8, 8) - 1797 examples with 8 sets of 8 features each (square image, 8x8 pixels)

# # View an example image from digits
# import matplotlib.pyplot as plt
# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)

# Reshape the digits data to be 2d
data = digits.images.reshape((digits.images.shape[0], -1))
print(data.shape) # (1797, 64) - reduced to 1797 examples with 64 features each