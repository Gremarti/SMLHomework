from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

import pandas
from pandas import DataFrame

import os

workspace = "/home/toshuumilia/Workspace/SML/"  # Insert the working directory here.
datasetPath = workspace + "data/sms.tsv"  # Tells where is located the data
experimentOnePath = workspace + "experiment/experimentOne.csv"  # Location of the first experiment result


smsDF = pandas.read_table(datasetPath, header=None, names=["label", "message"])
smsDF["label_numerical"] = smsDF.label.map({"ham": 0, "spam": 1})

smsDataset = smsDF.message
smsLabel = smsDF.label_numerical

methodArray = []
measureArray = []
valueArray = []

# Simulate ten trees so we can have an average.
for x in range(0, 15):
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
    decisionTree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                          max_features=None, random_state=None, max_leaf_nodes=None,
                                          min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
                                          presort=False)
    decisionTree.fit(trainDTM, label_train)
    
    label_predicted = decisionTree.predict(testDTM)
    
    # SEE: https://en.wikipedia.org/wiki/Precision_and_recall
    valueArray.append(metrics.precision_score(label_test, label_predicted))
    measureArray.append("precision")
    methodArray.append("Decision Tree")

    valueArray.append(metrics.recall_score(label_test, label_predicted))
    measureArray.append("recall")
    methodArray.append("Decision Tree")

    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    measureArray.append("accuracy")
    methodArray.append("Decision Tree")

    valueArray.append(metrics.f1_score(label_test, label_predicted))
    measureArray.append("f1score")
    methodArray.append("Decision Tree")
    
    # LOGISTIC REGRESSION
    # TODO: Explore which parameters could be used.
    # SEE: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    logisticRegression = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                                            C=1.0, fit_intercept=True, intercept_scaling=1,
                                            class_weight=None, random_state=None, solver='liblinear',
                                            max_iter=100, multi_class='ovr', verbose=0,
                                            warm_start=False, n_jobs=1)
    logisticRegression.fit(trainDTM, label_train)
    
    label_predicted = logisticRegression.predict(testDTM)
    
    valueArray.append(metrics.precision_score(label_test, label_predicted))
    measureArray.append("precision")
    methodArray.append("Logistic Regression")

    valueArray.append(metrics.recall_score(label_test, label_predicted))
    measureArray.append("recall")
    methodArray.append("Logistic Regression")

    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    measureArray.append("accuracy")
    methodArray.append("Logistic Regression")

    valueArray.append(metrics.f1_score(label_test, label_predicted))
    measureArray.append("f1score")
    methodArray.append("Logistic Regression")
    
    # NEURAL NETWORK
    # SEE: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    neuralNetwork = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='adam',
                                  alpha=0.0001, batch_size='auto', learning_rate='constant',
                                  learning_rate_init=0.001, power_t=0.5, max_iter=200,
                                  shuffle=True, random_state=None, tol=0.0001,
                                  verbose=False, warm_start=False, momentum=0.9,
                                  nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                  beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    neuralNetwork.fit(trainDTM, label_train)
    
    label_predicted = neuralNetwork.predict(testDTM)

    valueArray.append(metrics.precision_score(label_test, label_predicted))
    measureArray.append("precision")
    methodArray.append("Neural Network")

    valueArray.append(metrics.recall_score(label_test, label_predicted))
    measureArray.append("recall")
    methodArray.append("Neural Network")

    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    measureArray.append("accuracy")
    methodArray.append("Neural Network")

    valueArray.append(metrics.f1_score(label_test, label_predicted))
    measureArray.append("f1score")
    methodArray.append("Neural Network")

    print("Step", x, "done.")
    
experimentOneDF = DataFrame()
experimentOneDF["measure"] = measureArray
experimentOneDF["value"] = valueArray
experimentOneDF["method"] = methodArray

if not os.path.exists(workspace + "results/"):
    os.makedirs(workspace + "results/")
    
experimentOneDF.to_csv(experimentOnePath)
