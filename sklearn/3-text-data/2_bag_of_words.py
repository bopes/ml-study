# Bag of Words
# This algorithm creates a different object for each word in every document, resulting :
# key = document id/index
# value = count for that word in the given document

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

# Drawback 1:
# This approach creates a different object for every word across all documents. This is prohibitely expensive

# Solution 1:
# Add 'sparsity'. This means that only non-zero counts will be included for each word and stopwords will be excluded (AKA ignore boring words)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))

# Drawback 2:
# This approach may have difficulty handling long documents with high word counts, even if they share topics with shorter documents

# Solution 2:
# Consider the frequency of word appearances rather than a normal count. This is called Term Frequency (tf)

# Drawback 3:
# Interesting words that appear in almost every document are not particularly revealing. Words that appear in a smaller number of documents are more revealing.

# Solution 3:
# Weight words inversely to the number of documents they appear in. This is called Inverse Document Frequency (idf)

# Both tf and idf can be applied with a 'tdidf' transformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) # fit_transform both fits the tranformer to the parameter data and returns transformed data
print(X_train_tfidf.shape)
# X_train_tdidf represents our list of features

# With our features set, we can begin training a classifier
# The Naive Bayes classifier is a good baseline for text analysis. The multinomial NB varient is appropriate for word counts
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Let's test the classifier with some dummy data
docs_new = ['God is love','OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
  print("%r => %s" % (doc, twenty_train.target_names[category]))