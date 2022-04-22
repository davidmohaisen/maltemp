import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

f = open("../Pickles/MalwareRespreading/RevivalChainsInfo","rb")
data_set,family_set,dates_set,md5_set = pickle.load(f)

counterChain = 0
for i in range(len(data_set)):
    # if len(family_set[i]) != 1 and len(list(set(family_set[i]))) != 1  and len(list(set(dates_set[i]))) != 1:
    if len(family_set[i]) != 1  and len(list(set(dates_set[i]))) != 1:
        counterChain += 1
        print("Revival Chain",str(counterChain)+":")
        for j in range(len(family_set[i])):
            print("\t","Appeared as",family_set[i][j],"in",dates_set[i][j],"with MD5 of:",md5_set[i][j])




for i in range(len(data_set)):
    if len(family_set[i]) != 1  and len(list(set(dates_set[i]))) >= 5:
        counterChain += 1
        fam_chain = []
        date_chain = []
        MD5_chain = []

        for j in range(len(family_set[i])):
            if dates_set[i][j] not in date_chain:
                fam_chain.append(family_set[i][j])
                date_chain.append(dates_set[i][j])
                MD5_chain.append(md5_set[i][j])
        # Convert date strings (e.g. 2014-10-18) to datetime
        dates = [datetime.strptime(d, "%m/%d/%y") for d in date_chain]
        date_chain = [d.strftime("%m/%d/%Y") for d in dates]
        levels = np.tile([4,-4, 3,-3, 2,-2],
                         int(np.ceil(len(dates)/6)))[:len(dates)]

        # Create figure and plot a stem plot with the date
        fig, ax = plt.subplots(figsize=(14,4), constrained_layout=True)
        print(fam_chain)
        print(MD5_chain)
        colors = ["g"]
        for i in range(len(fam_chain)-1):
            if fam_chain[i+1] == fam_chain[i]:
                colors.append("b")
            else:
                colors.append("r")
        # colors = ["b"]+(["tab:red"]*len(dates[1:]))
        ax.vlines(dates, 0, levels, color=colors)  # The vertical stems.
        # ax.vlines(dates[:1], 0, levels[:1], color="b")  # The vertical stems.
        # ax.vlines(dates[1:], 0, levels[1:], color="tab:red")  # The vertical stems.

        ax.plot(dates, np.zeros_like(dates), "-o",
                color="k", markerfacecolor="k")  # Baseline and markers on it.


        vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
        rot = np.array([90, 90])[(levels > 0).astype(int)]
        marginY = np.array([-2, 2])[(levels > 0).astype(int)]
        for d, l, r, va, rt,mY in zip(dates, levels, date_chain, vert,rot,marginY):
            ax.annotate(r, xy=(d, l), xytext=(0, mY),
                        textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=16)

        # format xaxis with 4 month intervals
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # remove y axis and spines
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.margins(y=0.1)
        plt.show()
        # exit()
