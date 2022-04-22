import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def getSamplesDates(MalwareList):
    dates = []
    md5 = []
    for d in MalwareList:
        dates.append(d[5].split(" ")[0])
        md5.append(d[0].split("_")[1])
    return dates,md5


f = open("../Pickles/Data","rb")
MalwareList = pickle.load(f)
Dates,md5 = getSamplesDates(MalwareList)


f = open("../Pickles/MalwareRespreading/EmberDataFiltered","rb")
data,family = pickle.load(f)

counter = 0
data_set = []
family_set = []
dates_set = []
md5_set = []
for i in range(len(data)):
    counter += 1
    print("Currently at",counter)
    if data[i].tolist() not in data_set:
        data_set.append(data[i].tolist())
        family_set.append([family[i]])
        dates_set.append([Dates[i]])
        md5_set.append([md5[i]])
    else:
        index = data_set.index(data[i].tolist())
        family_set[index].append(family[i])
        dates_set[index].append(Dates[i])
        md5_set[index].append(md5[i])


f = open("../Pickles/MalwareRespreading/RevivalChainsInfo","wb")
pickle.dump([data_set,family_set,dates_set,md5_set],f)

print(len(data_set))
exit()
