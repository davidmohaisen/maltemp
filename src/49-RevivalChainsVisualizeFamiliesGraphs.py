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

f = open("../Pickles/MalwareRespreading/RevivalChainsInfo","rb")
data_set,family_set,dates_set,md5_set = pickle.load(f)

edges = []
uniqueFamInChains = []

for i in range(len(data_set)):
    if len(family_set[i]) != 1:
        fam_chain = []
        for j in range(len(family_set[i])):
            if family_set[i][j] not in fam_chain  and family_set[i][j] != "SINGLETON":
                fam_chain.append(family_set[i][j])

        if len(fam_chain) > 1:
            # print(fam_chain)
            for subset in itertools.combinations(fam_chain, 2):
                if subset not in edges:
                    edges.append(subset)
            uniqueFamInChains+=fam_chain

nodes = list(set(uniqueFamInChains))
while True:
    figure(figsize=(8, 6))
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=None, pos=None, fixed=None, iterations=10, threshold=0.00001, weight='weight', scale=1, center=None, dim=2, seed=None)
    # pos = nx.kamada_kawai_layout(G, dist=None, pos=None, weight='weight', scale=1, center=None, dim=2)

    #https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spiral_layout.html
    nx.draw_networkx_edges(G, pos, alpha=0.9, width=1, edge_color="k")
    nx.draw_networkx_nodes(G, pos, node_size=750, node_color="#210070", alpha=0.0)
    label_options = {"ec": "k", "fc": "white", "alpha": 0.9}
    nx.draw_networkx_labels(G, pos, font_size=10, bbox=label_options)
    ax = plt.gca()
    # ax.margins(0.1, 0.05)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
