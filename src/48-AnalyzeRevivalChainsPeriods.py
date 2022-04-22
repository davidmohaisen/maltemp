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


periods = []
allMalware = 0
for i in range(len(data_set)):
    if len(family_set[i]) != 1:
        fam_chain = []
        date_chain = []
        for j in range(len(family_set[i])):
            allMalware += 1
            if dates_set[i][j] not in date_chain:
                fam_chain.append(family_set[i][j])
                date_chain.append(dates_set[i][j])
        dates = [datetime.strptime(d, "%m/%d/%y") for d in date_chain]
        if abs((dates[0] - dates[-1]).days) > -1:
            periods.append(abs((dates[0] - dates[-1]).days))


print(max(periods))
print(sum(periods)/len(periods))
print(len(periods))
print(allMalware)
