import numpy as np
import os
path ='H:/Ender/boresight_calibration/20210617_swir/swir/raw/100426_20210617_bsc_uvvs307_2021_06_17_23_21_55'
flist=[]
for i in os.listdir(path):
    if i.startswith('raw'):
        if i.endswith('.hdr') or i.endswith('_nuc'):
            continue
        else:
            flist.append(i+'.txt')
for i in flist:
    with open(i,'w') as f:
        f.write("#,x,y")
