# Real life test case
# We will identify photo subjects by studying other photos of them

# Display  progress logs
from time import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Import data and convert to np.arrays
# Import the photos
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# introspect the image arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
# We will use the data directly (ignoring relative pixel position)
X = lfw_people.data
n_features = X.shape[1]
# The label to predict person (target results)
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
# Print Raw Data Status Check
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d\n" % n_classes)

# Split into train/test sets using stritified KFold
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Use PCA to reduce dimensionality and create 'eignefaces'
from sklearn.decomposition import PCA
n_components = 150 # the desired number of features
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
print("Done in %.3fs\n" % (time()-t0))
print("Reducing the input data")
t0 = time()
X_train_pca = pca.transform(X_train) # reduce original data  (train)
X_test_pca = pca.transform(X_test) # reduce original data (test)
print("Done in %.3fs\n" % (time()-t0))

# Train SVM classifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
print("Training classifier")
t0 = time()
param_grid = {'C': [1e3,5e3,1e4,5e4,1e5],
              'gamma': [0.0001,0.0005,0.001,0.005,0.01,0.1]} # use these options for training
clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'),param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Done in %.3fs\n" % (time() - t0))
print("Best estimator found:")
print(clf.best_estimator_)
print()

# Test the classifier
print("Predicting people's names on the test set")
t0 = time()
pred = clf.predict(X_test_pca)
print("Done in %.3fs\n" % (time() - t0))
# Print test results
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print("Classification Report:\n")
print(classification_report(y_test,pred,target_names=target_names))
print("Confusion matrix:\n")
print(confusion_matrix(y_test,pred,labels=range(n_classes)))
print()

# Display the test
import matplotlib.pyplot as plt

# Plot a gallery of portraits
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
  plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
  plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.9, hspace=.35)
  for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i+1)
    plt.imshow(images[i].reshape(h,w), cmap=plt.cm.gray)
    plt.title(titles[i],size=12)
    plt.xticks(())
    plt.yticks(())
# Plot the result of the prediction
def title(y_pred,y_test,target_names,i):
  pred_name = target_names[pred[i]].rsplit(' ',1)[-1]
  true_name = target_names[y_test[i]].rsplit(' ',1)[-1]
  return "predicted: %s\ntrue: %s" % (pred_name, true_name)
# Collect the  titles
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
prediction_titles = [title(pred, y_test,target_names, i) for i in range(y_test.shape[0])]
# Plot the results
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()
plot_gallery(X_test, prediction_titles, h, w)
plt.show()
