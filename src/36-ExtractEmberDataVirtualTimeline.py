import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import ember
f = open("../Pickles/VirtualTimeline/VirtualLineSamplesList","rb")
files,Fam = pickle.load(f)

extractor = ember.features.PEFeatureExtractor(2)


data = []
countFam = 0
for famFiles in files:
    data.append([])
    countFam += 1
    countFiles = 0
    for file in famFiles:
        countFiles += 1
        print(countFam, countFiles,len(famFiles))
        try:
            bin = open(file, "rb").read()
        except:
            print("file not found",file)
        features = np.array(extractor.feature_vector(bin), dtype=np.float32)
        data[-1].append(features)
    data[-1] = np.asarray(data[-1])

print("Done")
for d in data:
    print(d.shape)
f = open("../Pickles/VirtualTimeline/EmberData","wb")
pickle.dump([data,Fam],f)
