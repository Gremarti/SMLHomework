import matplotlib.pyplot as pyplot
import seaborn

import pandas

workspace = "/home/toshuumilia/Workspace/SML/"  # Insert the working directory here.
datasetPath = workspace + "data/sms.tsv"  # Tells where is located the data
experimentOnePath = workspace + "results/experimentOne.csv"  # Location of the first experiment result
globalFigsize = (15, 6)  # Graphs parameters

experimentOneDF = pandas.read_csv(experimentOnePath)

seaborn.set_style("whitegrid")
pyplot.figure(figsize=globalFigsize)
seaborn.barplot(x="measure", y="value", hue="method",
                data=experimentOneDF, palette="Blues_d")
pyplot.ylabel('value', fontsize=12)
pyplot.xlabel('measure', fontsize=12)
pyplot.title('Insert Title', fontsize=15)
pyplot.show()
