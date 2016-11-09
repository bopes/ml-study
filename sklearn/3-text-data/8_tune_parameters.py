from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)
docs_test = twenty_test.data

text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=5,random_state=42)),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# Set up Grid Search to refine parameters
from sklearn.model_selection import GridSearchCV
# These are the parameters to test
parameters = {'vect__ngram_range': [(1,1),(1,2)],
              'tfidf__use_idf': (True,False),
              'clf__alpha': (1e-2,1e-3),
             }
# Refine classifier
gs_clf = GridSearchCV(text_clf,parameters,n_jobs=-1)
# Refit newly refined classifier to the data
gs_clf.fit(twenty_train.data,twenty_train.target)
# Test refined results
prediction = gs_clf.predict(docs_test)
# Take a look at the accuracy
print(np.mean(prediction == twenty_test.target))