
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime


f= open("../Pickles/OnlineRatios","rb")

DetectedRatio, DetectionPerEngineRatio = pickle.load(f)


yearStart = 2010
yearEnd = 2020

tprMonthly = []
tprMonthlyPerEngine = []
LabelString = []
indexLabel = []
for i in range(len(DetectedRatio)):
    if len(DetectedRatio[i]) < 10:
        continue
    if i%12 == 0:
        indexLabel.append(i)
        yearL = yearStart+(i//12)
        WeekL = (i%12)+1
        d = str(yearL)+'-'+str(WeekL)
        r = datetime.datetime.strptime(d, "%Y-%m")
        LabelString.append(r.strftime("%Y"))


    tpr = 1.0*sum(DetectedRatio[i])/len(DetectedRatio[i])
    print("Month:",i,"TPR",tpr)
    tprMonthly.append(tpr)
    tpr = 1.0*sum(DetectionPerEngineRatio[i])/len(DetectionPerEngineRatio[i])
    print("Month:",i,"TPR",tpr)
    tprMonthlyPerEngine.append(tpr)





x = np.linspace(0, len(tprMonthlyPerEngine)-1, len(tprMonthlyPerEngine))
z = np.polyfit(x, tprMonthlyPerEngine, 5)
p = np.poly1d(z)
y = p(np.arange(0,len(tprMonthlyPerEngine)))
plt.plot(x,y,"k-",linewidth=2,label='TPR')
margin = y-tprMonthlyPerEngine
plt.fill_between(x, y , y-margin, color="k", alpha=0.33)

plt.xticks(labels = LabelString, ticks = indexLabel)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Performance", fontsize=18)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.ylim(0.00, 1.00)
# plt.xlim(0, 102)
plt.xlim(0, 126)
plt.grid()
plt.legend(loc='lower right',  fontsize=14)

plt.show()
