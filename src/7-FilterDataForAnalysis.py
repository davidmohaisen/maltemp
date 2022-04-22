import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
from shutil import copyfile

f = open("../Pickles/DataFiltered","rb")
data = pickle.load(f)
for d in data:
    try:
        copyfile("../Data/Win/Malware/"+d[0], "../Data/Win/FilteredMalware/"+d[0] )
    except:
        continue
