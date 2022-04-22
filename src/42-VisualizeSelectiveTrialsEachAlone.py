import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import pickle

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

f = open("../Pickles/VirtualTimeline/VirtualSimulationResults2","rb")
Metrics_List,Actual_Distribution,Predicted_Distribution = pickle.load(f)
Fam = ["zbot","playtech","bladabindi","gamarue","webdialer","fareit","razy","ursu","darkkomet","emotet"]

# for i in range(len(Actual_Distribution)):
#     print(i,":",Metrics_List[i][0],Metrics_List[i][1],Metrics_List[i][2])
#     print("\t",Actual_Distribution[i])
#     print("\t",Predicted_Distribution[i])
# exit()
takenIndeces = [3,61,200]

namesActual = []
datesActual = []
colorActual = []
namesPredicted = []
datesPredicted = []
colorPredicted = []

for taken in takenIndeces:
    namesActual.append([])
    datesActual.append([])
    namesPredicted.append([])
    datesPredicted.append([])
    colorActual.append([])
    colorPredicted.append([])
    print("Trial #",taken)
    for key in Actual_Distribution[taken].keys():
        try:
            print("\t",Fam[key],"Actual:",Actual_Distribution[taken][key],"Predicted:",Predicted_Distribution[taken][key])
            namesActual[-1].append(Fam[key])
            datesActual[-1].append(Actual_Distribution[taken][key])
            namesPredicted[-1].append(Fam[key])
            datesPredicted[-1].append(Predicted_Distribution[taken][key])
            colorActual[-1].append("blue")
            colorPredicted[-1].append("crimson")

        except:
            print("\t",Fam[key],"Actual:",Actual_Distribution[taken][key],"Not Predicted")
            namesActual[-1].append(Fam[key])
            datesActual[-1].append(Actual_Distribution[taken][key])
            colorActual[-1].append("m")



for i_index in range(3):
    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(constrained_layout=True,figsize=(14,4))
    # Choose some nice levels
    levels = np.tile([4, 3, 2],
                     int(np.ceil(len(datesActual[i_index])/3)))[:len(datesActual[i_index])]
    ax.vlines(datesActual[i_index], 0, levels, color=colorActual[i_index])  # The vertical stems.
    ax.plot(datesActual[i_index], np.zeros_like(datesActual[i_index]), "-o",
            color="k", markerfacecolor="k")  # Baseline and markers on it.

    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    rot = np.array([90, 90])[(levels > 0).astype(int)]
    marginY = np.array([-2, 2])[(levels > 0).astype(int)]
    for d, l, r, va, rt,mY in zip(datesActual[i_index], levels, namesActual[i_index], vert,rot,marginY):
        ax.annotate(r, xy=(d, l), xytext=(0, mY),
                    textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=16)


    # Choose some nice levels
    levels = np.tile([-4, -3, -2],
                     int(np.ceil(len(datesPredicted[i_index])/3)))[:len(datesPredicted[i_index])]
    ax.vlines(datesPredicted[i_index], 0, levels, color=colorPredicted[i_index])  # The vertical stems.
    ax.plot(datesPredicted[i_index], np.zeros_like(datesPredicted[i_index]), "-o",
            color="k", markerfacecolor="k")  # Baseline and markers on it.

    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    rot = np.array([90, 90])[(levels > 0).astype(int)]
    marginY = np.array([-2, 2])[(levels > 0).astype(int)]
    for d, l, r, va, rt,mY in zip(datesPredicted[i_index], levels, namesPredicted[i_index], vert,rot,marginY):
        ax.annotate(r, xy=(d, l), xytext=(0, mY),
                    textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=16)



    # remove y axis and spines
    ax.yaxis.set_visible(False)
    if i_index != 2:
        ax.xaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i_index != 2:
        ax.spines["bottom"].set_visible(False)
    ax.margins(y=0.3)
    if i_index != 2:
        ax.axis(ymin=-4,ymax=4)
    else:
        ax.axis(ymin=-12,ymax=4)

    if i_index == 2:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(2))

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.xlabel("Steps", fontsize=18)
        plt.xticks(fontsize=16)
    plt.show()



exit()
