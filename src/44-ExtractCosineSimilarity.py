import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics.pairwise import cosine_similarity


f = open("../Pickles/MalwareRespreading/EmberData","rb")
data,family = pickle.load(f)

counter = 0
data_set = []
for d in data:
    counter += 1
    print("Currently at",counter)
    if d.tolist() not in data_set:
        data_set.append(d.tolist())
print(len(data_set))
exit()

cos_data = cosine_similarity(data)
cos_data[cos_data>1] = 1
cos_data[cos_data<0] = 0

print(cos_data.min(),cos_data.max())
# cos_data = (cos_data) / (cos_data.max()-cos_data.min())

np.fill_diagonal(cos_data, 0)
print(cos_data.shape)



for i in range(len(cos_data)):
    # if len(cos_data[i][cos_data[i]>0.9999]) != 0:
    # print(cos_data[i].max())
    print(len(cos_data[i][cos_data[i]>=1.0]))
