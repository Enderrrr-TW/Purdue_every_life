import gdal
import matplotlib.pyplot as plt 

import numpy as np
import os
from pandas import read_csv as read_csv
path='H:/Ender/banding/20210512/targets/nhs199_gryfn_1546'
driver = gdal.GetDriverByName('GTIFF')
t=read_csv('H:/Ender/banding/bands.txt',names=['0','1'],header=None)
t['1'][0]=398.631
bands=t['1']

target_list=[]
for i in os.listdir(path):
    if i.endswith('.hdr') or i.endswith('.xml'):
        continue

    target_list.append(i)
# print(target_list)
for target in target_list:
    fname=path+'/'+target
    print(fname)
    img=gdal.Open(fname)
    X=img.RasterXSize
    Y=img.RasterYSize
    img=img.ReadAsArray()
    # print(img.shape)
    # print(X,Y)
    # img=[band,Y,X]
    x_list=[] # gradient: num of positive - num of negative
    y_list=[]
    mean_gradient_x=[]
    mean_gradient_y=[]
    for j in range(img.shape[0]):
        
        gradient=np.array(np.gradient(img[j]))
        # result = driver.Create("E:/t100/regression/t90_scipy_trf_Crimmins.tiff", xsize = ncol, ysize = nrow, bands=1,eType = gdal.GDT_Float32)
        # result.GetRasterBand(1).WriteArray(new_image)
        # plt.imshow(gradient)
        size=gradient.size # num of pixels
        mean_gradient_y.append(np.mean(gradient[0]))
        mean_gradient_x.append(np.mean(gradient[1]))
        gradient[0][gradient[0]>0]=1
        gradient[0][gradient[0]<0]=-1
        gradient[1][gradient[1]>0]=1
        gradient[1][gradient[1]<0]=-1
        positive_y=np.sum([gradient[0]>=0])
        positive_x=np.sum([gradient[1]>=0])
        negative_y=np.sum([gradient[0]<0])
        negative_x=np.sum([gradient[1]<0])

        x_list.append((positive_x)/size)
        y_list.append((positive_y)/size)
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    plt.title('Y direction')
    plt.ylim(0,1)
    plt.plot(bands,y_list,label='positive rate')
    plt.subplot(222)
    plt.title('X direction')
    plt.ylim(0,1)
    plt.plot(bands,x_list,label='positive rate')
        # plt.imshow(gradient[0])
    plt.subplot(223)
    plt.plot(bands,mean_gradient_y)
    plt.subplot(224)
    plt.title(target)

    plt.plot(bands,mean_gradient_x)
    # plt.legend()
    # plt.show()
    plt.savefig('H:/Ender/banding/20210512/nhs199_gryfn_1546/output_targets_An-Te/'+target+'.png')
    # quit()
