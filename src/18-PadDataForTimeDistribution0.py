import matplotlib.pyplot as plt
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import os,binascii

base = "../Data/Win/FilteredMalware/"
baseToSave = "../Data/Win/testSamples/"

onlyfiles = [f for f in listdir(base) if isfile(join(base, f))]


f = open("../Pickles/DataFiltered","rb")
MalwareList = pickle.load(f)



counter = 0

copyfile(base+"VirusShare_7f66ce02bd73120982250acbd6633cd5", baseToSave+"VirusShare_7f66ce02bd73120982250acbd6633cd5")
copyfile(base+"VirusShare_7f66ce02bd73120982250acbd6633cd5", baseToSave+"VirusShare_7f66ce02bd73120982250acbd6633cd5"+"Padded")
f = open(baseToSave+"VirusShare_7f66ce02bd73120982250acbd6633cd5"+"Padded","ab")
f.write(bytearray("0xff", 'utf-8'))
