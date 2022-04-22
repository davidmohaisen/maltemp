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
takenIndeces = [3,10,200]

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


# exit()
# names = ['v2.2.4', 'v3.0.3', 'v3.0.2', 'v3.0.1', 'v3.0.0', 'v2.2.3',
#          'v2.2.2', 'v2.2.1']
#
# dates = [1, 2, 3, 4,
#          5, 6, 7, 8]



# Create figure and plot a stem plot with the date
fig, ax = plt.subplots(nrows=3, gridspec_kw={'height_ratios': [1, 1, 2.5]}, constrained_layout=True)

# Choose some nice levels
levels = np.tile([4, 3, 2],
                 int(np.ceil(len(datesActual[0])/3)))[:len(datesActual[0])]
ax[0].vlines(datesActual[0], 0, levels, color=colorActual[0])  # The vertical stems.
ax[0].plot(datesActual[0], np.zeros_like(datesActual[0]), "-o",
        color="k", markerfacecolor="k")  # Baseline and markers on it.

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(datesActual[0], levels, namesActual[0], vert,rot,marginY):
    # ax[0].annotate(r, xy=(d, l), xytext=(0, mY),
    ax[0].annotate(r, xy=(d-0.8, 0.5), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=24)


# Choose some nice levels
levels = np.tile([-4, -3, -2],
                 int(np.ceil(len(datesPredicted[0])/3)))[:len(datesPredicted[0])]
ax[0].vlines(datesPredicted[0], 0, levels, color=colorPredicted[0])  # The vertical stems.
ax[0].plot(datesPredicted[0], np.zeros_like(datesPredicted[0]), "-o",
        color="k", markerfacecolor="k")  # Baseline and markers on it.

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(datesPredicted[0], levels, namesPredicted[0], vert,rot,marginY):
    # ax[0].annotate(r, xy=(d, l), xytext=(0, mY),
    ax[0].annotate(r, xy=(d-0.8, -0.5), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=24)



# remove y axis and spines
ax[0].yaxis.set_visible(False)
ax[0].xaxis.set_visible(False)
ax[0].spines["left"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["bottom"].set_visible(False)
ax[0].margins(y=0.3)
ax[0].axis(ymin=-4,ymax=4)














# Choose some nice levels
levels = np.tile([4, 3, 2],
                 int(np.ceil(len(datesActual[1])/3)))[:len(datesActual[1])]
ax[1].vlines(datesActual[1], 0, levels, color=colorActual[1])  # The vertical stems.
ax[1].plot(datesActual[1], np.zeros_like(datesActual[1]), "-o",
        color="k", markerfacecolor="k")  # Baseline and markers on it.

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(datesActual[1], levels, namesActual[1], vert,rot,marginY):
    ax[1].annotate(r, xy=(d, l), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=24)


# Choose some nice levels
levels = np.tile([-4, -3, -2],
                 int(np.ceil(len(datesPredicted[1])/3)))[:len(datesPredicted[1])]
ax[1].vlines(datesPredicted[1], 0, levels, color=colorPredicted[1])  # The vertical stems.
ax[1].plot(datesPredicted[1], np.zeros_like(datesPredicted[1]), "-o",
        color="k", markerfacecolor="k")  # Baseline and markers on it.

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(datesPredicted[1], levels, namesPredicted[1], vert,rot,marginY):
    ax[1].annotate(r, xy=(d, l), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=24)



# remove y axis and spines
ax[1].yaxis.set_visible(False)
ax[1].xaxis.set_visible(False)
ax[1].spines["left"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["bottom"].set_visible(False)
ax[1].margins(y=0.3)
ax[1].axis(ymin=-5,ymax=5)



# Choose some nice levels
levels = np.tile([4, 3, 2],
                 int(np.ceil(len(datesActual[2])/3)))[:len(datesActual[2])]
ax[2].vlines(datesActual[2], 0, levels, color=colorActual[2])  # The vertical stems.
ax[2].plot(datesActual[2], np.zeros_like(datesActual[2]), "-o",
        color="k", markerfacecolor="k")  # Baseline and markers on it.

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(datesActual[2], levels, namesActual[2], vert,rot,marginY):
    ax[2].annotate(r, xy=(d, l), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=24)


# Choose some nice levels
levels = np.tile([-4, -3, -2],
                 int(np.ceil(len(datesPredicted[2])/3)))[:len(datesPredicted[2])]
ax[2].vlines(datesPredicted[2], 0, levels, color=colorPredicted[2])  # The vertical stems.
ax[2].plot(datesPredicted[2], np.zeros_like(datesPredicted[2]), "-o",
        color="k", markerfacecolor="k")  # Baseline and markers on it.

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(datesPredicted[2], levels, namesPredicted[2], vert,rot,marginY):
    ax[2].annotate(r, xy=(d, l), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=24)



# remove y axis and spines
ax[2].yaxis.set_visible(False)
ax[2].spines["left"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)
ax[2].margins(y=0.3)



ax[2].xaxis.set_major_locator(MultipleLocator(10))
ax[2].xaxis.set_minor_locator(MultipleLocator(2))
ax[2].axis(ymin=-30,ymax=5)

plt.setp(ax[2].get_xticklabels(), rotation=30, ha="right")
plt.xlabel("Steps", fontsize=30)
plt.xticks(fontsize=24)
plt.show()







# ax[0].plot(np.arange(0,1000), l_step_error, '-r', label="SDR",lw=0.2);
# ax[0].plot(np.arange(0,1000), l_false_acceptance, '-k', label="FAR",lw=0.2);
# ax[0].plot(np.arange(0,1000), l_average_detection, '-b', label="ADR",lw=0.2);
# ax[0].xaxis.set_major_locator(MultipleLocator(200))
# ax[0].xaxis.set_minor_locator(MultipleLocator(20))
# ax[0].yaxis.set_major_locator(MultipleLocator(0.20))
# ax[0].yaxis.set_minor_locator(MultipleLocator(0.020))
# ax[0].grid(True, linestyle='--')
# ax[0].axis(xmin=0,xmax=1000,ymin=0,ymax=1)
# ax[0].get_xaxis().set_visible(False)
#
# ax[1].plot(np.arange(0,1000), l_step_error, '-r', label="SDR",lw=0.2);
# ax[1].plot(np.arange(0,1000), l_false_acceptance, '-k', label="FAR",lw=0.2);
# ax[1].plot(np.arange(0,1000), l_average_detection, '-b', label="ADR",lw=0.2);
# ax[1].xaxis.set_major_locator(MultipleLocator(200))
# ax[1].xaxis.set_minor_locator(MultipleLocator(20))
# ax[1].yaxis.set_major_locator(MultipleLocator(0.20))
# ax[1].yaxis.set_minor_locator(MultipleLocator(0.020))
# ax[1].grid(True, linestyle='--')
# ax[1].axis(xmin=0,xmax=1000,ymin=0,ymax=1)
# ax[1].get_xaxis().set_visible(False)
#
#
#
# ax[2].plot(np.arange(0,1000), l_step_error, '-r', label="SDR",lw=0.2);
# ax[2].plot(np.arange(0,1000), l_false_acceptance, '-k', label="FAR",lw=0.2);
# ax[2].plot(np.arange(0,1000), l_average_detection, '-b', label="ADR",lw=0.2);
# ax[2].xaxis.set_major_locator(MultipleLocator(200))
# ax[2].xaxis.set_minor_locator(MultipleLocator(20))
# ax[2].yaxis.set_major_locator(MultipleLocator(0.20))
# ax[2].yaxis.set_minor_locator(MultipleLocator(0.020))
# ax[2].grid(True, linestyle='--')
# ax[2].axis(xmin=0,xmax=1000,ymin=0,ymax=1)
#
# plt.show()
