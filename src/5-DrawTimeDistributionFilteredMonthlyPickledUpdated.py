import pickle
import matplotlib as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
f = open("../Pickles/Dist08-20", "rb")
Date = pickle.load(f)
DataHisto = []
for i in range(len(Date)):
    for j in range(Date[i]):
        DataHisto.append(i)
DateX = []
ticksVis = []
for i in range(len(Date)):
    if i % 24 == 0:
        DateX.append(str(2008 + int(i / 12)))
        ticksVis.append(i)
with plt.style.context(['science', 'grid']):
    fig, ax = plt.subplots()
    p1 = plt.hist(DataHisto,bins=len(Date),linewidth=1,edgecolor = '#1a237c', color="#224392")
    #p2 = plt.bar(ind+0.2, womenMeans, width,linewidth=1, edgecolor = '#1a237c', color = "#d2d9e9")
    plt.ylabel('Number of Samples',fontsize=12)
    plt.xlabel('Time (months)', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(xmin=0, xmax=len(Date))
    plt.xticks(labels=DateX, ticks=ticksVis)
    plt.xlabel("Time (Months)", fontsize=12)
    # plt.legend(['$\mathcal{T}_{K}$', '$\mathcal{T}_{A}$'])
plt.margins(0,0)
plt.savefig('../Figures/ahmad.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0)
