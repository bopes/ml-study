from sklearn import svm, datasets

# Create classifier
clf = svm.SVC()

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train classifier
clf.fit(X, y)


# Import joblib to persist classifier to new file
from sklearn.externals import joblib

# Save classifier specs to new file
joblib.dump(clf, 'iris_clf/iris_clf.pkl')

# Load classifer from file
clf2 = joblib.load('iris_clf/iris_clf.pkl')


# # Import pickle to persist classifier
# import pickle

# # Save classifier specs to pickle
# s = pickle.dumps(clf)

# # Load classiier specs to new classifier instance
# clf2 = pickle.load(s)


# Use new classifier to make prediction
pred = clf2.predict(X[0:1])

# Display prediction
print(pred)