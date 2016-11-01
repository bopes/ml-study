from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()

print("Data")
print(digits.data)
print("Target")
print(digits.target)
print("original sample")
print(digits.images[0])
