import hashlib
print(hashlib.md5(open('../Data/Win/FilteredMalware/VirusShare_a95111407437bd851ae651f847b53e90','rb').read()).hexdigest())
print(hashlib.md5(open('../Data/Win/FilteredMalwarePadded/VirusShare_a95111407437bd851ae651f847b53e90','rb').read()).hexdigest())
