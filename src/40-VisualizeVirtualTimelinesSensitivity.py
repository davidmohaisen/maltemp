import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
import scipy as sp
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.pyplot import figure
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
f = open("../Pickles/VirtualTimeline/VirtualSimulationSensitivity","rb")
Metrics_List,Actual_Distribution,Predicted_Distribution = pickle.load(f)

l_false_acceptance = []
l_average_detection = []
l_step_error = []
for m in Metrics_List:
    l_false_acceptance.append(m[0])
    l_average_detection.append(m[1])
    l_step_error.append(m[2]/100)

l_false_acceptance.append(1.0)
l_average_detection.append(1.0)
l_step_error.append(0.0)
# l_false_acceptance = l_false_acceptance[::-1]
# l_average_detection = l_average_detection[::-1]
# l_step_error = l_step_error[::-1]

fig, ax = plt.subplots()
fig.set_size_inches(3,3, forward=True)

ax.plot(np.arange(0,1.01,0.01), l_false_acceptance, '-k', label="FAR");
ax.plot(np.arange(0,1.01,0.01), l_average_detection, '-b', label="ADR");
ax.plot(np.arange(0,1.01,0.01), l_step_error, '-r', label="SDR");
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.02))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))
plt.ylim(0.00, 1.0)
plt.xlim(0.0, 1)
plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)
plt.xlabel("Sensitivity", fontsize=16)
plt.ylabel("Performance", fontsize=16)
plt.grid(True, linestyle='--')
plt.legend(loc="lower right",fontsize=14)
# plt.legend(loc='upper center',fontsize=14, bbox_to_anchor=(0.5, 1.30),
          # fancybox=True, ncol=3)
plt.show()
