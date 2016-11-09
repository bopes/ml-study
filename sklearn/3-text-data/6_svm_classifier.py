from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
# Get train data
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)
# Create Classifier
text_clf = Pipeline([
  ('vect',CountVectorizer()),
  ('tfidf',TfidfTransformer()),
  ('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=5,random_state=42)), # Try an SVM classifier
])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
# Get test data
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)
docs_test = twenty_test.data
# Make prediction
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target)) # returns 0.913...