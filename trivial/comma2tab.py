import os
import pandas as pd
path='H:/Ender/boresight_calibration/20210617_swir/swir/processed'
output='H:/Ender/boresight_calibration/20210617_swir/swir/processed/img_obs'
fp=os.listdir(path)
# gcp position lock
# I only put the target at 0 or 0.5
def gcp_position_lock(num):
    d=num-int(num)
    if d >0.5:
        if d>0.75:
            num=int(num)+1
        else:
            num=int(num)+0.5
    elif d<0.5:
        if d<0.25:
            num=int(num)
        else:
            num=int(num)+0.5
    return num
for i in fp:
    if i.endswith('.txt'):
        csv=pd.read_csv(path+'/'+i)
        for j in range(len(csv['x'])):
            csv['x'][j]=gcp_position_lock(csv['x'][j])
        for j in range(len(csv['y'])):
            csv['y'][j]=gcp_position_lock(csv['y'][j])
        csv.to_csv(output+'/'+i,sep='\t', index=False)

# for i in fp:
#     if i.endswith('.txt'):
#         with open(path+'/'+i,'r') as f:
#             text=f.read()
#             text=text.replace(',','\t')
#         with open(output+'/'+i,'w') as f:
#             f.write(text)
