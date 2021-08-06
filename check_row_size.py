import laspy
from numba import jit,njit, types, vectorize
import pandas as pd
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon ,Polygon

import matplotlib.pyplot as plt
import random
import collections
def check_size(old,new,threshold):
    index=np.abs(new['x4']-new['x1'])<threshold
    new['x1'][index]=old['x1'][index]
    new['y1'][index]=old['y1'][index]
    new['x2'][index]=old['x2'][index]
    new['y2'][index]=old['y2'][index]
    new['x3'][index]=old['x3'][index]
    new['y3'][index]=old['y3'][index]
    new['x4'][index]=old['x4'][index]
    new['y4'][index]=old['y4'][index]
    return new
def create_shp(rec,fname):
    p_lsit=[]
    for i in range(len(rec['x1'])):
        p_lsit.append(Polygon([(rec['x1'][i],rec['y1'][i]),(rec['x2'][i],rec['y2'][i]),(rec['x3'][i],rec['y3'][i]),(rec['x4'][i],rec['y4'][i])]))
    shp=MultiPolygon(p_lsit)
    features=[i for i in range(len(rec['x1']))] # shp ID=0-49
    f=gpd.GeoDataFrame({'feature':features,'geometry':shp})
    f.to_file(fname)
rec=pd.read_csv('rec2.csv')
rec=dict(rec)
result=pd.read_csv('./result/all_raw/q03_008_10cm.csv')
result=dict(result)
result_modified=check_size(rec,result,0.7)
fname2=f'./result/all_raw/q03_008_10cm_modified_70cm.shp'
create_shp(result_modified,fname2)
