import os
flist=os.listdir('H:/Ender/boresight_calibration/20210513_india/vnir/processed/img_obs')
for i in flist:
    with open(i,'r') as f:
        text=f.read()
        if ' ' in text:
            print(i)
    