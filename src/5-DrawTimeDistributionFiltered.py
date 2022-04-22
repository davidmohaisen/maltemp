import matplotlib.pyplot as plt
import pickle
import numpy as np
f = open("../Pickles/DataFiltered","rb")
data = pickle.load(f)

startYear = data[0][2]
startWeek = data[0][4]
lastYear = data[-1][2]
lastWeek = data[-1][4]

Date = []
for i in range(lastYear-startYear+1):
    start = 0
    end = 53
    if i == lastYear-startYear:
        end = lastWeek
    for j in np.arange(start,end):
        Date.append(0)
for d in data:
    year = d[2]
    week = d[4]
    location = ((year-startYear)*53) + (week-1)
    # print(year,week,location)
    Date[location] += 1




DataHisto = []
for i in range(len(Date)):
    for j in range(Date[i]):
        DataHisto.append(i)

DateX = []
ticksVis = []
for i in range(len(Date)):
    if i%106 == 0:
        DateX.append(str(2006+int(i/53)))
        ticksVis.append(i)



plt.hist(DataHisto,bins=len(Date),color="#666666",lw=0,ec="black")
# plt.ylim(ymin=0, ymax = 1000)
plt.xlim(xmin=0, xmax = len(Date))
plt.xticks(labels = DateX, ticks = ticksVis)
plt.xlabel("Time (weekly)", fontsize=18)
plt.ylabel("# samples", fontsize=18)

plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.show()

for i in range(2020-2006+1):
    print(2006+i,sum(Date[i*53:(i+1)*53]))
