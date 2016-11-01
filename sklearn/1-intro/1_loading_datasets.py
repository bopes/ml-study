# Import datasets from sklearn
from sklearn import datasets

# Import specific datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

# View some data
digits.data # raw data
digits.target # known results for how the data should be classified