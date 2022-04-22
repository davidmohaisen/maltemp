import requests
import json
import os
import subprocess
import cv2
import numpy
import pickle
import array
import os
import pickle
import os
import scipy
import array
import numpy
import scipy.misc
import imageio
from PIL import Image

names = []
path = "../Data/Win/Benign/"
pathG = "../Data/Win/Images/Pickle/Benign/"
for sample in os.listdir(path):
    names.append(sample)

f = open("../Pickles/BenignList","wb")
pickle.dump(names,f)
f.close()
