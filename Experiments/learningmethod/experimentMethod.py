from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

import pandas
from pandas import DataFrame

import os

workspace = "/home/toshuumilia/Workspace/SML/"  # Insert the working directory here.
datasetPath = workspace + "data/sms.tsv"  # Tells where is located the data

if not os.path.exists(workspace + "results/"):
    os.makedirs(workspace + "results/")

smsDF = pandas.read_table(datasetPath, header=None, names=["label", "message"])
smsDF["label_numerical"] = smsDF.label.map({"ham": 0, "spam": 1})

smsDataset = smsDF.message
smsLabel = smsDF.label_numerical

methodArray = []
measureArray = []
valueArray = []

availableMeasures = ["Precision", "Recall", "Accuracy", "F1Score"]
availableMethods = ["Decision Tree", "Logistic Regression", "Neural Network", "Naive Bayesian"]

# Simulate ten trees so we can have an average.
for x in range(0, 10):
    # Create the datasets and the labels used for the ML.
    # TODO: Parameter to test: how to split the smsDataset into train and test.
    dataset_train, dataset_test, label_train, label_test = train_test_split(smsDataset, smsLabel, random_state=1)
    
    # Note: DTM=documentTermMatrix
    vectorizer = CountVectorizer()
    trainDTM = vectorizer.fit_transform(dataset_train)
    testDTM = vectorizer.transform(dataset_test)
    
    # DECISION TREE
    # TODO: Explore which parameters could be used.
    # SEE: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    decisionTree = DecisionTreeClassifier()
    decisionTree.fit(trainDTM, label_train)
    
    label_predicted = decisionTree.predict(testDTM)
    
    # SEE: https://en.wikipedia.org/wiki/Precision_and_recall
    valueArray.append(metrics.precision_score(label_test, label_predicted))
    valueArray.append(metrics.recall_score(label_test, label_predicted))
    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    valueArray.append(metrics.f1_score(label_test, label_predicted))
    
    for index in range(0, 4):
        measureArray.append(availableMeasures[index])
        methodArray.append(availableMethods[0])
    
    # LOGISTIC REGRESSION
    # TODO: Explore which parameters could be used.
    # SEE: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    logisticRegression = LogisticRegression()
    logisticRegression.fit(trainDTM, label_train)
    
    label_predicted = logisticRegression.predict(testDTM)

    valueArray.append(metrics.precision_score(label_test, label_predicted))
    valueArray.append(metrics.recall_score(label_test, label_predicted))
    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    valueArray.append(metrics.f1_score(label_test, label_predicted))

    for index in range(0, 4):
        measureArray.append(availableMeasures[index])
        methodArray.append(availableMethods[1])
    
    # NEURAL NETWORK
    # SEE: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    neuralNetwork = MLPClassifier()
    
    neuralNetwork.fit(trainDTM, label_train)
    
    label_predicted = neuralNetwork.predict(testDTM)

    valueArray.append(metrics.precision_score(label_test, label_predicted))
    valueArray.append(metrics.recall_score(label_test, label_predicted))
    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    valueArray.append(metrics.f1_score(label_test, label_predicted))

    for index in range(0, 4):
        measureArray.append(availableMeasures[index])
        methodArray.append(availableMethods[2])
    
    # NAIVE BAYESIAN
    # SEE: http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    naiveBayesian = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

    naiveBayesian.fit(trainDTM, label_train)

    label_predicted = naiveBayesian.predict(testDTM)

    valueArray.append(metrics.precision_score(label_test, label_predicted))
    valueArray.append(metrics.recall_score(label_test, label_predicted))
    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    valueArray.append(metrics.f1_score(label_test, label_predicted))

    for index in range(0, 4):
        measureArray.append(availableMeasures[index])
        methodArray.append(availableMethods[3])

    print("Step", x, "done.")
    
experimentBasicMethodsDF = DataFrame()
experimentBasicMethodsDF["Measure"] = measureArray
experimentBasicMethodsDF["Value"] = valueArray
experimentBasicMethodsDF["Method"] = methodArray

experimentBasicMethodsDF.to_csv(workspace + "results/experimentBasicMethods.csv")
