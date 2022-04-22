import pickle
f = open("../Pickles/DataFiltered","rb")
data = pickle.load(f)
famDist2019 = []
famDist = []

for d in data:
    if d[6] != "SINGLETON" and d[6] != "-":
        if d[2] >= 2019:
            famDist2019.append(d[6])
        elif d[2] >= 2017 and d[2] <= 2018:
            famDist.append(d[6])

famDist2019 = list(set(famDist2019))
famDist = list(set(famDist))

print(len(famDist2019),len(famDist))

fam = set(famDist2019).intersection(set(famDist))
print(len(famDist2019)-len(fam))
