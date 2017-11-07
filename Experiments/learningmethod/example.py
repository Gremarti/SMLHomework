from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
import pandas

# Example of dataset

iris = load_iris()

featureMatrix = iris.data
labelVector = iris.target

print(featureMatrix.shape)
print(labelVector.shape)
print(iris.feature_names)
print(featureMatrix[1])

# ----------------------#
# Text training example #
# ----------------------#

print("---")

dataset = ["call you tonight", "Call me a cab", "please call me... PLEASE!"]
vector = CountVectorizer()

# Learn the "vocabulary" of the training data (occurs in-place)
vector.fit(dataset)
featureNames = vector.get_feature_names()

print(featureNames)

# Transform training data into a "document-term matrix'
documentTermMatrix = vector.transform(dataset)

# convert sparse matrix to a dense matrix
documentTermMatrix.toarray()

# examine the vocabulary and document-term matrix together
df = pandas.DataFrame(documentTermMatrix.toarray(), columns=vector.get_feature_names())
print(df.head())
