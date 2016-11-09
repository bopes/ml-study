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
