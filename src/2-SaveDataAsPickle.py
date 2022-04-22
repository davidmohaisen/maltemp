import matplotlib.pyplot as plt
import pickle
f = open("/media/ahmed/HDD/MalwareTemporalRobustness/Data/Win/AV3.csv","r")
data = []
lines = f.readlines()[1:]
for line in lines:
    line = line.replace("\n","")
    line = line.replace("\"","")
    line = line.split(";")
    line[0] = "VirusShare_"+line[0]
    line[2] = int(float(line[2]))
    line[3] = int(float(line[3]))
    line[4] = int(float(line[4]))
    data.append(line)

f = open("../Pickles/Data","wb")
pickle.dump(data,f)
