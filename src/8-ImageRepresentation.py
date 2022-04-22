import os
import pickle
import os
import scipy
import array
import numpy
import scipy.misc
import imageio
from PIL import Image

count = 1
path = "../Data/Win/Benign/"
pathG = "../Data/Win/Images/Benign/"
for sample in os.listdir(path):
    try:
        file_to_scan = path+sample
        f = open(file_to_scan,'rb');
        ln = os.path.getsize(file_to_scan);
        width = 256;
        rem = ln%width;
        a = array.array("B");
        a.fromfile(f,ln-rem);
        f.close();
        g = numpy.reshape(a,(int(len(a)/width),width));
        g = numpy.uint8(g);
        g = Image.fromarray(g.astype('uint8'))
        g = g.resize((256,256));
        g.save(pathG+sample+'.png')
    except:
        print("passed benign")
path = "../Data/Win/FilteredMalware/"
pathG = "../Data/Win/Images/Malware/"
for sample in os.listdir(path):
    try:
        file_to_scan = path+sample
        f = open(file_to_scan,'rb');
        ln = os.path.getsize(file_to_scan);
        width = 256;
        rem = ln%width;
        a = array.array("B");
        a.fromfile(f,ln-rem);
        f.close();
        g = numpy.reshape(a,(int(len(a)/width),width));
        g = numpy.uint8(g);
        g = Image.fromarray(g.astype('uint8'))
        g = g.resize((256,256));
        g.save(pathG+sample+'.png')
    except:
        print("passed malware")
