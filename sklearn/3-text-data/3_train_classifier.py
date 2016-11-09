from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# With our features set, we can begin training a classifier
# The Naive Bayes classifier is a good baseline for text analysis. The multinomial NB varient is appropriate for word counts
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Let's test the classifier with some dummy data
# Create dummy docs
docs_new = ['God is love','OpenGL on the GPU is fast']
# Transform the dummy docs
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# Predict classes for dummy docs
predicted = clf.predict(X_new_tfidf)
# Print the results
for doc, category in zip(docs_new, predicted):
  print("%r => %s" % (doc, twenty_train.target_names[category]))