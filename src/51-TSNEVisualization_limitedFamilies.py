import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

#
# f = open("../Pickles/MalwareRespreading/EmberDataFiltered","rb")
# data_,family_ = pickle.load(f)
# Fam = ['regotet', 'jinto', 'icedid', 'graz', 'ymacco', 'qbot', 'ligooc', 'zusy', 'gozi', 'guloader', 'autinject', 'installerex', 'zbot', 'netwalker', 'vebzenpak', 'ryuk', 'ursnif', 'csfrsys', 'trickbot', 'mansabo', 'cerbu', 'genkryptik', 'fareit', 'boaxxe', 'zenpak', 'torrentlocker', 'terkcop', 'atraps', 'dynamer', 'emotet', 'deyma', 'cutwail']
# data = []
# family = []
# emotetCounter = 0
# for i in range(len(data_)):
#     if family_[i] in Fam:
#         if family_[i] == "emotet" and emotetCounter == 1000:
#             continue
#         elif family_[i] == "emotet":
#             emotetCounter+= 1
#         data.append(data_[i])
#         family.append(family_[i])
# unique_families = list(set(family))
#
# print(len(data))
# tsne = TSNE(n_components=2,perplexity=100,n_jobs=-1)
# z = tsne.fit_transform(data)
#
# f = open("../Pickles/DistributionVitualization/TSNEVisualizationSamplesLimitedFamilies","wb")
# pickle.dump([z,family,unique_families],f)

f = open("../Pickles/DistributionVitualization/TSNEVisualizationSamplesLimitedFamilies","rb")
z,family,unique_families = pickle.load(f)

df = pd.DataFrame()
df["y"] = family
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(unique_families)),
                data=df)
# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", len(unique_families)),
#                 data=df, legend = False)
plt.axis('off')

plt.show()
