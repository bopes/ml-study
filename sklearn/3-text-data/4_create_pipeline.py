from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

# Create a pipeline to make the classifier easier to work with
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
  ('vect',  CountVectorizer()  ),
  ('tfidf', TfidfTransformer() ),
  ('clf',   MultinomialNB()    ),
])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)