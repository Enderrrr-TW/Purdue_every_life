import gdal
import numpy as np
import os
from numba import njit, types, vectorize, prange

'''
Binning bands is to get the average of two bands in hyperspectral image
'''

@njit()
def bin2bands(img):
    '''
    This function will reduce the bands to half of the original number
    if a band is left, I give up that band
    '''
    bands=img.shape[0]//2
    binned_img=np.empty((bands,img.shape[1],img.shape[2]),dtype=np.float64)
    tempp_img=np.empty((img.shape[1],img.shape[2]),dtype=np.float64)
    b=0
    for i in range(img.shape[0]):
        if i%2==0:
            print(i)
            tempp_img=(img[i]+img[i+1])/2
            # d=int(i/2)
            binned_img[b]=tempp_img
            b=b+1
    print(np.mean(binned_img[132]))
    return binned_img
    
driver=gdal.GetDriverByName('GTIFF')
path='H:/Ender/banding/20210605/20200605_banding_india_afternoon/100386_20200605NH1990227_2021_06_05_18_27_47'
flist=[]
for i in os.listdir(path):
    if i.endswith('.txt') or i.endswith('.hdr') or i.endswith('json') or i.endswith('result'):
        continue
    else:
        flist.append(i)
print(flist)

for i in flist:
    img=gdal.Open((path+'/'+i))
    [col,row] = [img.RasterXSize, img.RasterYSize]
    img=img.ReadAsArray()
    result=bin2bands(img)
    print(np.mean(result[132]))
    # quit()
    fname=path+'/binned_result/'+i+'binned.tiff'
    new_img = driver.Create(fname,col,row,bands=result.shape[0],eType = gdal.GDT_Float32)
    for j in prange(result.shape[0]):
        bb=j+1
        new_img.GetRasterBand(bb).WriteArray(result[j])
