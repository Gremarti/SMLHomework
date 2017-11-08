import matplotlib.pyplot as pyplot
import seaborn

import pandas

workspace = "/home/toshuumilia/Workspace/SML/"  # Insert the working directory here.
datasetPath = workspace + "data/sms.tsv"  # Tells where is located the data

# Experiment location

# Graphs parameters
globalFigsize = (12, 6)


# Comparison Experiment #
#
# experimentOneDF = pandas.read_csv(experimentOnePath)
#
# seaborn.set_style("darkgrid")
# pyplot.figure(figsize=globalFigsize)
# seaborn.barplot(x="Value", y="Measure", hue="Method",
#                 data=experimentOneDF)
# pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# pyplot.ylabel('Measure', fontsize=12)
# pyplot.xlabel('Value', fontsize=12)
# pyplot.xlim(0.5, 1)
# pyplot.title('Performance comparison between four learning methods', fontsize=15)
# pyplot.show()


# Decision Tree #
# Depth Experiment
#
# experimentDTDepthDF = pandas.read_csv(workspace + "results/experimentDTDepth.csv")
#
# seaborn.set_style("whitegrid")
# pyplot.figure(figsize=globalFigsize)
# seaborn.barplot(x="Value", y="Measure", hue="Depth",
#                 data=experimentDTDepthDF)
# pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# pyplot.ylabel('Measure', fontsize=12)
# pyplot.xlabel('Value', fontsize=12)
# pyplot.xlim(0.5, 1)
# pyplot.title('Performance comparison of a Decision Tree relative to a maximum depth', fontsize=15)
# pyplot.show()

# Criterion Experiment
#
# experimentDTCriterionDF = pandas.read_csv(workspace + "results/experimentDTCriterion.csv")
#
# seaborn.set_style("whitegrid")
# pyplot.figure(figsize=globalFigsize)
# seaborn.barplot(x="Value", y="Measure", hue="Criterion",
#                 data=experimentDTCriterionDF)
# pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# pyplot.ylabel('Measure', fontsize=12)
# pyplot.xlabel('Value', fontsize=12)
# pyplot.xlim(0.5, 1)
# pyplot.title('Performance comparison of a Decision Tree relative to a splitting quality criterion', fontsize=15)
# pyplot.show()

# MinSampleSplit Experiment
#
# experimentDTMinSampleSplitDF = pandas.read_csv(workspace + "results/experimentDTMinSampleSplit.csv")
#
# seaborn.set_style("whitegrid")
# pyplot.figure(figsize=globalFigsize)
# seaborn.barplot(x="Value", y="Measure", hue="MinSampleSplit",
#                 data=experimentDTMinSampleSplitDF)
# pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# pyplot.ylabel('Measure', fontsize=12)
# pyplot.xlabel('Value', fontsize=12)
# pyplot.xlim(0.5, 1)
# pyplot.title('Insert Title', fontsize=15)
# pyplot.xticks(rotation='vertical')
# pyplot.show()

# MaxFeature Experiment
#
# experimentDTMaxFeatureDF = pandas.read_csv(workspace + "results/experimentDTMaxFeature.csv")
#
# seaborn.set_style("whitegrid")
# pyplot.figure(figsize=globalFigsize)
# seaborn.barplot(x="Value", y="Measure", hue="MaxFeature",
#                 data=experimentDTMaxFeatureDF)
# pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# pyplot.ylabel('Measure', fontsize=12)
# pyplot.xlabel('Value', fontsize=12)
# pyplot.xlim(0.5, 1)
# pyplot.title('Insert Title', fontsize=15)
# pyplot.xticks(rotation='vertical')
# pyplot.show()

# MaxLeafNodes Experiment

# experimentDTMaxLeafNodesDF = pandas.read_csv(workspace + "results/experimentDTMaxLeafNodes.csv")
#
# seaborn.set_style("whitegrid")
# pyplot.figure(figsize=globalFigsize)
# seaborn.pointplot(y="Value", hue="Measure", x="MaxLeafNodes",
#                 data=experimentDTMaxLeafNodesDF)
# pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# pyplot.ylabel('Measure', fontsize=12)
# pyplot.xlabel('Value', fontsize=12)
# pyplot.xlim(0.5, 1)
# pyplot.title('Insert Title', fontsize=15)
# pyplot.xticks(rotation='vertical')
# pyplot.show()

# MinImpurityDecrease Experiment
#
# experimentDTMinImpurityDecreaseDF = pandas.read_csv(workspace + "results/experimentDTMinImpurityDecrease.csv")
#
# seaborn.set_style("whitegrid")
# pyplot.figure(figsize=globalFigsize)
# seaborn.pointplot(y="Value", hue="Measure", x="MinImpurityDecrease",
#                   data=experimentDTMinImpurityDecreaseDF, palette="Greens_d")
# pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# pyplot.ylabel('Measure Value', fontsize=12)
# pyplot.xlabel('Min Impurity Decrease', fontsize=12)
# pyplot.title('', fontsize=15)
# pyplot.xticks(rotation='vertical')
# pyplot.show()

# BasicVsOptimized Experiment

experimentDTBasicVsOptimizedDF = pandas.read_csv(workspace + "results/experimentDTBasicVsOptimized.csv")

seaborn.set_style("whitegrid")
pyplot.figure(figsize=globalFigsize)
seaborn.barplot(x="Value", y="Measure", hue="Tuning",
                data=experimentDTBasicVsOptimizedDF)
pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pyplot.ylabel('Measure', fontsize=12)
pyplot.xlabel('Value', fontsize=12)
pyplot.xlim(0.5, 1)
pyplot.title('Insert Title', fontsize=15)
pyplot.xticks(rotation='vertical')
pyplot.show()
