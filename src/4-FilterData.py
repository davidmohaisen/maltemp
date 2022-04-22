import matplotlib.pyplot as plt
import pickle
import numpy as np
f = open("../Pickles/Data","rb")
data = pickle.load(f)

DataFiltered = []
CurrentTime = ""
CurrentFamilies = []
CountFamilies = []
Buffer = []


for i in range(len(data)):
    tmpTime = str(data[i][2])+"-"+str(data[i][4])
    if CurrentTime != tmpTime:
        print(CurrentTime)
        if len(Buffer) <= 200:
            DataFiltered = DataFiltered + Buffer
        else :
            CurrentFamilies = list(set(CurrentFamilies))
            count = 0
            SingCount = 0
            taken = []
            while count < 200:
                oldCount = count
                CountFamilies = [0]*len(CurrentFamilies)
                for j in range(len(Buffer)):
                    if Buffer[j] in taken:
                        continue
                    if Buffer[j][6] == "SINGLETON":
                        if SingCount < 100:
                            taken.append(Buffer[j])
                            SingCount += 1
                    elif CountFamilies[CurrentFamilies.index(Buffer[j][6])] < 1:
                        CountFamilies[CurrentFamilies.index(Buffer[j][6])] += 1
                        taken.append(Buffer[j])
                        count += 1
                if oldCount == count:
                    break
            DataFiltered = DataFiltered + taken
        CurrentTime = tmpTime
        CurrentFamilies = []
        Buffer = []
    else:
        CurrentFamilies.append(data[i][6])
        Buffer.append(data[i])

f = open("../Pickles/DataFiltered","wb")
pickle.dump(DataFiltered,f)
