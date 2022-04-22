import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

f = open("../Pickles/DataFiltered","rb")
data = pickle.load(f)
famDist = {}
for d in data:
    if d[6] != "SINGLETON" and d[6] != "-":
        if d[6] not in famDist.keys():
            famDist[d[6]] = 1
        else:
            famDist[d[6]] += 1
names = []
dates = []
reqY = [2014,2015]
for d in data:
    if d[6] != "SINGLETON" and d[6] != "-" and d[6] not in names and (famDist[d[6]] >58 or (d[2] in reqY and famDist[d[6]] > 20)) :
        names.append(d[6])
        dates.append(d[5].split(" ")[0])

# Convert date strings (e.g. 2014-10-18) to datetime
dates = [datetime.strptime(d, "%m/%d/%y") for d in dates]

# Choose some nice levels
levels = np.tile([100,-100, 75, -75, 50, -50],
                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date
fig, ax = plt.subplots(figsize=(18, 3))
# ax.set(title="Matplotlib release dates")

markerline, stemline, baseline = ax.stem(dates, levels,
                                         linefmt="C3-", basefmt="k-")


plt.setp(markerline, mec="k", mfc="k", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.
markerline.set_ydata(np.zeros(len(dates)))

# annotate lines
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
rot = np.array([90, 90])[(levels > 0).astype(int)]
marginY = np.array([-2, 2])[(levels > 0).astype(int)]
for d, l, r, va, rt,mY in zip(dates, levels, names, vert,rot,marginY):
    ax.annotate(r, xy=(d, l), xytext=(0, mY),
                textcoords="offset points", rotation=rt, va=va, ha="center",fontsize=8.5)

# format xaxis with 4 month intervals
ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=9))
ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# remove y axis and spines
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)
plt.ylim(ymin=-475, ymax = 250)
ax.margins(y=0.1)
plt.show()
