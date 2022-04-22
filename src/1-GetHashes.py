import os
import hashlib
f = open("../Data/Win/MalwareResources.csv","w")
for root, dirs, files in os.walk("../Data/Win/Malware/"):
    for filename in files:        
        f.write(filename+","+(hashlib.md5(open("../Data/Win/Malware/"+filename,'rb').read()).hexdigest())+"\n")
exit()
