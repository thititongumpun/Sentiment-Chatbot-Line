import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pd.Series([65, 61, 25, 22, 27]).plot(kind="bar")

plotdata = pd.DataFrame(
    {"pies": [10, 10, 42, 17, 37]}, 
    index=["Dad", "Mam", "Bro", "Sis", "Me"])
# Plot a bar chart
plotdata.plot(kind="bar")
plotdata['pies'].plot(kind="bar", title="test")
# Rotate the x-labels by 30 degrees, and keep the text aligned horizontally
plt.xticks(rotation=30, horizontalalignment="center")
plt.title("Mince Pie Consumption Study Results")
plt.xlabel("Family Member")
plt.ylabel("Pies Consumed")

plotdata = pd.DataFrame({
    "pies_2018":[40, 12, 10, 26, 36],
    "pies_2019":[19, 8, 30, 21, 38],
    "pies_2020":[10, 10, 42, 17, 37]
    }, 
    index=["Dad", "Mam", "Bro", "Sis", "Me"]
)
plotdata.head()

plotdata = pd.DataFrame({
    "Presicion":[0.8351, 0.8021, 0.8641],
    "Recall":[0.8332, 0.8049, 0.9052],
    "F1-Score":[0.8341, 0.8010, 0.8841]
    }, 
    index=["Logistic Regression", "Naive Bayes", "LSTM"]
)
ax = plotdata.plot(kind="bar")

plt.title("")
plt.xlabel("Compare Models", fontsize=18.5)
plt.ylabel("%", fontsize=18.5)
plt.xticks(rotation=0, horizontalalignment="center")

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1, p.get_height() * 0.95), rotation=0, fontsize=16.5)
    
fig = plt.gcf()
plt.rcParams.update({'font.size': 18})
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(top=1.1)
fig.set_size_inches((20, 7), forward=False)
plt.subplots_adjust(bottom=0.15)
fig.savefig("f1.png", dpi=500)
ax.legend(bbox_to_anchor=(0.99, 1.0))

plt.show()
