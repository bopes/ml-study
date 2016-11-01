from sklearn import datasets, svm

# Load data
iris = datasets.load_iris()
digits = datasets.load_digits()

# Create Classifier
clf = svm.SVC(gamma=0.001, C=100.)

# Fit classifier to data (all but last example)
clf.fit(digits.data[:-1], digits.target[:-1])

# Predict the last example from data
pred = clf.predict(digits.data[-1:])

# Display prediction
print(pred)