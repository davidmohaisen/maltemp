import os
import hashlib
f = open("../Data/Win/MalwareResources_OldData.csv","w")
for root, dirs, files in os.walk("../old/Data/Win/Malware/"):
    for filename in files:
        f.write(filename+","+(hashlib.md5(open("../old/Data/Win/Malware/"+filename,'rb').read()).hexdigest())+"\n")
exit()
