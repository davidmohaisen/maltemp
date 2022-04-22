import matplotlib.pyplot as plt
import pickle
import numpy as np



f = open("../Pickles/Dist08-20","rb")
Date = pickle.load(f)




DataHisto = []
for i in range(len(Date)):
    for j in range(Date[i]):
        DataHisto.append(i)

DateX = []
ticksVis = []
for i in range(len(Date)):
    if i%24 == 0:
        DateX.append(str(2008+int(i/12)))
        ticksVis.append(i)



plt.hist(DataHisto,bins=len(Date),color="#666666",lw=0,ec="black")
# plt.ylim(ymin=0, ymax = 1000)
plt.xlim(xmin=0, xmax = len(Date))
plt.xticks(labels = DateX, ticks = ticksVis)
plt.xlabel("Time (monthly)", fontsize=18)
plt.ylabel("# samples", fontsize=18)

plt.xlim(0, 152)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)

plt.show()

for i in range(2020-2008+1):
    print(2008+i,sum(Date[i*12:(i+1)*12]))
