import gdal
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
gcp_name=[]
gcp_path='H:/Ender/boresight_calibration/20210513_india/vnir/processed/img_obs'
# gcp_path='H:/Ender/boresight_calibration/20210513_india/vnir/processed/img_obs'
gcp_list=[]
for i in os.listdir(gcp_path):
    gcp_list.append(gcp_path+'/'+i)
    gcp_name.append(i)

img_path='H:/Ender/boresight_calibration/20210513_india/vnir/processed/rgb_or'
# img_path='H:/Ender/boresight_calibration/20210513_india/vnir/processed/rgb_or'
img_list=[]
img_name=[]
for i in os.listdir(img_path):
    # if i.endswith('.hdr'):
    #     continue
    # if i.startswith('raw'):
    #     img_list.append(img_path+'/'+i)
    #     img_name.append(i)
    if i.endswith('or'):
        img_list.append(img_path+'/'+i)
        img_name.append(i)   
print(img_name)
print(gcp_name)
if len(gcp_list)!=len(img_list):
    print('you fuck up')
    exit()
for i in range(len(gcp_list)):
    gcps=pd.read_csv(gcp_list[i],sep='\t')
    print(gcp_name[i])
    print(img_name[i])
    img=gdal.Open(img_list[i])
    img=img.GetRasterBand(1).ReadAsArray()
    # [B,G,R]=img.ReadAsArray()
    # R=R/(np.max(R)-np.min(R))
    # G=G/(np.max(G)-np.min(G))
    # B=B/(np.max(B)-np.min(B))
    # img=np.stack((B,G,R),axis=2)
    # plt.matshow(img)
    f=plt.figure()
    plt.imshow(img)
    for j in range(len(gcps['x'])):
        plt.scatter(gcps['x'][j],gcps['y'][j],color='red',s=5)
        plt.text(gcps['x'][j],gcps['y'][j]+50,color='red',s=gcps['#'][j])
    plt.title(img_name[i])
    plt.show()
    # plt.imsave('H:/Ender/boresight_calibration/20210513_india/vnir/processed/check_gcp/'+img_name[i]+'_check',f)
