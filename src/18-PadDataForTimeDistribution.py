import matplotlib.pyplot as plt
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import os,binascii

base = "../Data/Win/FilteredMalware/"
baseToSave = "../Data/Win/FilteredMalwarePadded/"

onlyfiles = [f for f in listdir(base) if isfile(join(base, f))]

for file in onlyfiles:
    copyfile(base+file, baseToSave+file)
    f = open(baseToSave+file,"ab")
    f.write(bytearray("0xff76483a858b1eff76483a85800b1eff76483a858b1eff76483a858b1eff76483a858b1eff76483a858b1eff76483a858b1eff76483a858b1ef00f76483a858b1eff76483a8580000b1eff76483a80058b1eff7648300a858b1eff76483a80058b1eff76483a858b1eff76483a858b1eff76483a858b1e", 'utf-8'))
