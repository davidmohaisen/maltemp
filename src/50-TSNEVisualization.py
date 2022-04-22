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

# f = open("../Pickles/MalwareRespreading/EmberDataFiltered","rb")
# data,family = pickle.load(f)
# data = data
# family = family
# unique_families = list(set(family))
#
# tsne = TSNE(n_components=2,perplexity=30,n_jobs=-1)
# z = tsne.fit_transform(data)

# f = open("../Pickles/DistributionVitualization/TSNEVisualizationSamples","wb")
# pickle.dump([z,family,unique_families],f)
f = open("../Pickles/DistributionVitualization/TSNEVisualizationSamples","rb")
z_,family_,_ = pickle.load(f)
z = []
family = []
for i in range(len(z_)):
    if family_[i] != "SINGLETON":
        family.append(family_[i])
        z.append(z_[i])
unique_families = list(set(family))
z = np.asarray(z)

df = pd.DataFrame()
df["y"] = family
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(unique_families)),
                data=df, legend = False)
plt.axis('off')

plt.show()
