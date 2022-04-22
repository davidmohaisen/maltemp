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

width = 64
path = "../Data/Win/Benign/"
pathG = "../Data/Win/Images/Pickle/Benign/"
for sample in os.listdir(path):
    try:
        file_to_scan = path+sample
        f = open(file_to_scan,'rb');
        ln = os.path.getsize(file_to_scan);
        rem = ln%width;
        a = array.array("B");
        a.fromfile(f,ln-rem);
        f.close();
        g = numpy.reshape(a,(int(len(a)/width),width));
        g = numpy.uint8(g);
        g = Image.fromarray(g.astype('uint8'))
        g = g.resize((width,width));
        g = numpy.array(g)

        f = open(pathG+sample,"wb")
        pickle.dump(g,f)
        f.close()
    except:
        print("passed benign")


path = "../Data/Win/FilteredMalware/"
pathG = "../Data/Win/Images/Pickle/Malware/"
for sample in os.listdir(path):
    try:
        file_to_scan = path+sample
        f = open(file_to_scan,'rb');
        ln = os.path.getsize(file_to_scan);
        rem = ln%width;
        a = array.array("B");
        a.fromfile(f,ln-rem);
        f.close();
        g = numpy.reshape(a,(int(len(a)/width),width));
        g = numpy.uint8(g);
        g = Image.fromarray(g.astype('uint8'))
        g = g.resize((width,width));
        g = numpy.array(g)

        f = open(pathG+sample,"wb")
        pickle.dump(g,f)
        f.close()
    except:
        print("passed Malware")
