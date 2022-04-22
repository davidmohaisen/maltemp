
import os
import scipy
import array
import numpy as np
import scipy.misc
import imageio
from PIL import Image
from sklearn.model_selection import train_test_split
import collections
import pickle
import numpy as np
import sys, os
from sklearn.decomposition import PCA
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.models import model_from_json
import datetime
from sklearn.ensemble import RandomForestClassifier
import json

def load_test_data_monthly(MalwareList,Detection, yearStart,yearEnd):
    maxDiff = []
    counter = 0
    for d in range(len(MalwareList)):
        if MalwareList[d][0]== "VirusShare_a95111407437bd851ae651f847b53e90" and Detection[d][1] ==0 or True:
            try:
                f = open("../Data/Win/v3reports/"+(MalwareList[d][0].replace("VirusShare_",""))+".json","r")
                data = json.load(f)
                if data["data"]["attributes"]["last_analysis_stats"]["malicious"]-Detection[d][1] >= 25 :
                    print(MalwareList[d],data["data"]["attributes"]["last_analysis_stats"]["malicious"],Detection[d][1])
                    maxDiff.append(data["data"]["attributes"]["last_analysis_stats"]["malicious"]-Detection[d][1])
            except:
                pass

    #     try:
    #         f = open("../Data/Win/v3reports/"+(MalwareList[d][0].replace("VirusShare_",""))+".json","r")
    #         data = json.load(f)
    #         if data["data"]["attributes"]["last_analysis_stats"]["malicious"]-Detection[d][1] < -5 and MalwareList[d][2] < 2019:
    #             print(MalwareList[d][0],MalwareList[d][2],data["data"]["attributes"]["last_analysis_stats"]["malicious"],Detection[d][1])
    #             counter += 1
    #         maxDiff.append(data["data"]["attributes"]["last_analysis_stats"]["malicious"]-Detection[d][1])
    #     except:
    #         pass
    #
    # maxDiff.sort()
    # print(maxDiff[:100])
    # print(maxDiff[-100:])
    # print(counter)






yearStart = 2012
yearEnd = 2020



f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)


f = open("../Pickles/PaddedReports","rb")
Detection = pickle.load(f)




load_test_data_monthly(MalwareList,Detection, yearStart,yearEnd)
