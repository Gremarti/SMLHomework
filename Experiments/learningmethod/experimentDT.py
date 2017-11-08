from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import pandas
from pandas import DataFrame

import numpy

import os

workspace = "/home/toshuumilia/Workspace/SML/"  # Insert the working directory here.
datasetPath = workspace + "data/sms.tsv"  # Tells where is located the data

smsCount = 5574

if not os.path.exists(workspace + "results/"):
    os.makedirs(workspace + "results/")

###################
# Loading dataset #
###################

smsDF = pandas.read_table(datasetPath, header=None, names=["label", "message"])
smsDF["label_numerical"] = smsDF.label.map({"ham": 0, "spam": 1})

smsDataset = smsDF.message
smsLabel = smsDF.label_numerical

methodArray = []
measureArray = []
valueArray = []
availableMeasures = ["Accuracy", "F1Score"]


dataset_train, dataset_test, label_train, label_test = train_test_split(smsDataset, smsLabel, random_state=1)

# Note: DTM=documentTermMatrix
vectorizer = CountVectorizer()
trainDTM = vectorizer.fit_transform(dataset_train)
testDTM = vectorizer.transform(dataset_test)

# DEPTH EXPERIMENT
# availableDepths = [None, 50, 25, 10, 5, 3]
#
# print("Depth Experiment")
# for x in range(0, 4):
#     for depth in availableDepths:
#         print("Step", x, "for depth:", depth)
#         # SEE: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#         decisionTree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=depth,
#                                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#                                               max_features=None, random_state=None, max_leaf_nodes=None,
#                                               min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
#                                               presort=False)
#         decisionTree.fit(trainDTM, label_train)
#
#         label_predicted = decisionTree.predict(testDTM)
#
#         # SEE: https://en.wikipedia.org/wiki/Precision_and_recall
#         valueArray.append(metrics.accuracy_score(label_test, label_predicted))
#         valueArray.append(metrics.f1_score(label_test, label_predicted))
#
#         for index in range(0, 2):
#             measureArray.append(availableMeasures[index])
#             methodArray.append("Depth-" + str(depth))
#
# # Save the experiments
# experimentDTDepthDF = DataFrame()
# experimentDTDepthDF["Measure"] = measureArray
# experimentDTDepthDF["Value"] = valueArray
# experimentDTDepthDF["Depth"] = methodArray
#
# experimentDTDepthDF.to_csv(workspace + "results/experimentDTDepth.csv")

# CRITERION EXPERIMENT
# availableCriterion = ["gini", "entropy"]
#
# methodArray = []
# measureArray = []
# valueArray = []
#
# print("Criteron Experiment")
# for x in range(0, 4):
#     for criterion in availableCriterion:
#         print("Step", x, "for criterion:", criterion)
#         decisionTree = DecisionTreeClassifier(criterion=criterion, splitter='best', max_depth=None,
#                                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#                                               max_features=None, random_state=None, max_leaf_nodes=None,
#                                               min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
#                                               presort=False)
#         decisionTree.fit(trainDTM, label_train)
#
#         label_predicted = decisionTree.predict(testDTM)
#
#         valueArray.append(metrics.accuracy_score(label_test, label_predicted))
#         valueArray.append(metrics.f1_score(label_test, label_predicted))
#
#         for index in range(0, 2):
#             measureArray.append(availableMeasures[index])
#             methodArray.append("Criterion-" + criterion)
#
# # Save the experiments
# experimentDTCriteronDF = DataFrame()
# experimentDTCriteronDF["Measure"] = measureArray
# experimentDTCriteronDF["Value"] = valueArray
# experimentDTCriteronDF["Criterion"] = methodArray
#
# experimentDTCriteronDF.to_csv(workspace + "results/experimentDTCriterion.csv")

# MIN_SAMPLES_SPLIT EXPERIMENT

# availableMinSampleSplit = [2, 10, 25, 50, 100, 250]
#
# methodArray = []
# measureArray = []
# valueArray = []
#
# print("MinSampleSplit Experiment")
# for x in range(0, 20):
#     for minSampleSplit in availableMinSampleSplit:
#         print("Step", x, "for minSampleSplit:", minSampleSplit)
#         decisionTree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
#                                               min_samples_split=minSampleSplit, min_samples_leaf=1,
#                                               min_weight_fraction_leaf=0.0,
#                                               max_features=None, random_state=None, max_leaf_nodes=None,
#                                               min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None,
#                                               presort=False)
#         decisionTree.fit(trainDTM, label_train)
#
#         label_predicted = decisionTree.predict(testDTM)
#
#         valueArray.append(metrics.accuracy_score(label_test, label_predicted))
#         valueArray.append(metrics.f1_score(label_test, label_predicted))
#
#         for index in range(0, 2):
#             measureArray.append(availableMeasures[index])
#             methodArray.append("MinSampleSplit-" + str(minSampleSplit))
#
# # Save the experiments
# experimentDTMinSampleSplitDF = DataFrame()
# experimentDTMinSampleSplitDF["Measure"] = measureArray
# experimentDTMinSampleSplitDF["Value"] = valueArray
# experimentDTMinSampleSplitDF["MinSampleSplit"] = methodArray
#
# experimentDTMinSampleSplitDF.to_csv(workspace + "results/experimentDTMinSampleSplit.csv")

# MAX_FEATURE EXPERIMENT

# availableMaxFeature = [None, "sqrt", "log2", 0.25, 0.5, 0.75]
#
# methodArray = []
# measureArray = []
# valueArray = []
#
# print("MaxFeature Experiment")
# for x in range(0, 10):
#     for maxFeature in availableMaxFeature:
#         print("Step", x, "for MaxFeature:", maxFeature)
#         decisionTree = DecisionTreeClassifier(max_features=maxFeature)
#         decisionTree.fit(trainDTM, label_train)
#
#         label_predicted = decisionTree.predict(testDTM)
#
#         valueArray.append(metrics.accuracy_score(label_test, label_predicted))
#         valueArray.append(metrics.f1_score(label_test, label_predicted))
#
#         for index in range(0, 2):
#             measureArray.append(availableMeasures[index])
#             methodArray.append("MaxFeature-" + str(maxFeature))
#
# # Save the experiments
# experimentDTMaxFeatureDF = DataFrame()
# experimentDTMaxFeatureDF["Measure"] = measureArray
# experimentDTMaxFeatureDF["Value"] = valueArray
# experimentDTMaxFeatureDF["MaxFeature"] = methodArray
#
# experimentDTMaxFeatureDF.to_csv(workspace + "results/experimentDTMaxFeature.csv")

# MAX_LEAF_NODES EXPERIMENT

# availableMaxLeafNodes = []
# for ratio in numpy.arange(1/6, 1.01, 1/6):
#     availableMaxLeafNodes.append(int(ratio * smsCount))

# availableMaxLeafNodes = numpy.concatenate([[2], numpy.arange(10, 270, 10)])

# methodArray = []
# measureArray = []
# valueArray = []
#
# print("MaxLeafNodes Experiment")
# for x in range(0, 5):
#     for maxLeafNodes in availableMaxLeafNodes:
#         print("Step", x, "for MaxLeafNodes:", maxLeafNodes)
#         decisionTree = DecisionTreeClassifier(max_leaf_nodes=maxLeafNodes)
#         decisionTree.fit(trainDTM, label_train)
#
#         label_predicted = decisionTree.predict(testDTM)
#
#         valueArray.append(metrics.accuracy_score(label_test, label_predicted))
#         valueArray.append(metrics.f1_score(label_test, label_predicted))
#
#         for index in range(0, 2):
#             measureArray.append(availableMeasures[index])
#             methodArray.append(maxLeafNodes)
#
# # Save the experiments
# experimentDTMaxLeafNodesDF = DataFrame()
# experimentDTMaxLeafNodesDF["Measure"] = measureArray
# experimentDTMaxLeafNodesDF["Value"] = valueArray
# experimentDTMaxLeafNodesDF["MaxLeafNodes"] = methodArray
#
# experimentDTMaxLeafNodesDF.to_csv(workspace + "results/experimentDTMaxLeafNodes.csv")

# MIN_IMPURITY_DECREASE

# availableMinImpurityDecrease = numpy.arange(0., 0.061, 0.005)
#
# methodArray = []
# measureArray = []
# valueArray = []
#
# print("MaxFeature Experiment")
# for x in range(0, 10):
#     for minImpurityDecrease in availableMinImpurityDecrease:
#         print("Step", x, "for MinImpurityDecrease:", minImpurityDecrease)
#         decisionTree = DecisionTreeClassifier(min_impurity_decrease=minImpurityDecrease)
#         decisionTree.fit(trainDTM, label_train)
#
#         label_predicted = decisionTree.predict(testDTM)
#
#         valueArray.append(metrics.accuracy_score(label_test, label_predicted))
#         valueArray.append(metrics.f1_score(label_test, label_predicted))
#
#         for index in range(0, 2):
#             measureArray.append(availableMeasures[index])
#             methodArray.append(str(minImpurityDecrease*100) + "%")
#
# # Save the experiments
# experimentDTMinImpurityDecreaseDF = DataFrame()
# experimentDTMinImpurityDecreaseDF["Measure"] = measureArray
# experimentDTMinImpurityDecreaseDF["Value"] = valueArray
# experimentDTMinImpurityDecreaseDF["MinImpurityDecrease"] = methodArray
#
# experimentDTMinImpurityDecreaseDF.to_csv(workspace + "results/experimentDTMinImpurityDecrease.csv")

# DEFAULT DT VS OPTIMIZED DT EXPERIMENT

availableMeasures = ["Precision", "Recall", "Accuracy", "F1Score"]
methodArray = []
measureArray = []
valueArray = []

print("MaxFeature Experiment")
for x in range(0, 20):
    print("Step", x, "for Basic Decision Tree")
    decisionTree = DecisionTreeClassifier()
    decisionTree.fit(trainDTM, label_train)

    label_predicted = decisionTree.predict(testDTM)

    valueArray.append(metrics.precision_score(label_test, label_predicted))
    valueArray.append(metrics.recall_score(label_test, label_predicted))
    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    valueArray.append(metrics.f1_score(label_test, label_predicted))

    for measure in availableMeasures:
        measureArray.append(measure)
        methodArray.append("Basic Decision Tree")

    print("Step", x, "for Custom Decision Tree")
    decisionTree = DecisionTreeClassifier(max_features=0.25, criterion="gini")
    decisionTree.fit(trainDTM, label_train)

    label_predicted = decisionTree.predict(testDTM)

    valueArray.append(metrics.precision_score(label_test, label_predicted))
    valueArray.append(metrics.recall_score(label_test, label_predicted))
    valueArray.append(metrics.accuracy_score(label_test, label_predicted))
    valueArray.append(metrics.f1_score(label_test, label_predicted))

    for measure in availableMeasures:
        measureArray.append(measure)
        methodArray.append("Custom Decision Tree")

# Save the experiments
experimentDTBasicVsOptimizedDF = DataFrame()
experimentDTBasicVsOptimizedDF["Measure"] = measureArray
experimentDTBasicVsOptimizedDF["Value"] = valueArray
experimentDTBasicVsOptimizedDF["Tuning"] = methodArray

experimentDTBasicVsOptimizedDF.to_csv(workspace + "results/experimentDTBasicVsOptimized.csv")
