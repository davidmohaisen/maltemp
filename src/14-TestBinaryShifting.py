import os
import pickle
import os
import scipy
import array
import numpy
import scipy.misc
import imageio
from PIL import Image

def binaryToText(a,path):
    f = open(path,"w")
    for i in range(len(a)):
        for j in range(len(a[i])):
            f.write(str(a[i][j])+" ")
        f.write("\n")


sample_1 = "../Data/Win/Malware/VirusShare_c88cba5c801efc41ef690e75572903b3"
sample_2 = "../Data/Win/Malware/VirusShare_c9569df1e16df16e0d9286320e9d1d0f"
sample_3 = "../Data/Win/Malware/VirusShare_097194f839632cdff6cb70de5df98c4a"
sample_4 = "../Data/Win/Malware/VirusShare_d90ce2bea547248ca6ddc8c72bfa28e5"

f = open(sample_1,'rb');
ln = os.path.getsize(sample_1);
width = 32;
rem = ln%width;
a = array.array("B");
a.fromfile(f,ln-rem);
f.close();
g = numpy.reshape(a,(int(len(a)/width),width));
g = numpy.uint8(g);
print(g.shape)
binaryToText(g,"../TXT/VirusShare_c88cba5c801efc41ef690e75572903b3.txt")


f = open(sample_2,'rb');
ln = os.path.getsize(sample_2);
width = 32;
rem = ln%width;
a = array.array("B");
a.fromfile(f,ln-rem);
f.close();
g = numpy.reshape(a,(int(len(a)/width),width));
g = numpy.uint8(g);
print(g.shape)
binaryToText(g,"../TXT/VirusShare_c9569df1e16df16e0d9286320e9d1d0f.txt")


f = open(sample_3,'rb');
ln = os.path.getsize(sample_3);
width = 32;
rem = ln%width;
a = array.array("B");
a.fromfile(f,ln-rem);
f.close();
g = numpy.reshape(a,(int(len(a)/width),width));
g = numpy.uint8(g);
print(g.shape)
binaryToText(g,"../TXT/VirusShare_097194f839632cdff6cb70de5df98c4a.txt")



f = open(sample_4,'rb');
ln = os.path.getsize(sample_4);
width = 32;
rem = ln%width;
a = array.array("B");
a.fromfile(f,ln-rem);
f.close();
g = numpy.reshape(a,(int(len(a)/width),width));
g = numpy.uint8(g);
print(g.shape)
binaryToText(g,"../TXT/VirusShare_d90ce2bea547248ca6ddc8c72bfa28e5.txt")
