from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=5,random_state=42)),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)
docs_test = twenty_test.data
prediction = text_clf.predict(docs_test)


from sklearn import metrics
# Classification report shows precision, recall, f1-score, support for each category
print(
  metrics.classification_report(
    twenty_test.target,
    prediction,
    target_names=twenty_test.target_names
  )
)
# Confusion matrix shows how often each category is mistaken for the others
print(
  metrics.confusion_matrix(
    twenty_test.target,
    prediction
  )
)