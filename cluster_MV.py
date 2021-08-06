'''
This is the python version of cluster_MV from Behrokh's code
Created by: An-Te Huang.
Date: 20210517
'''
import numpy as np
from scipy.spatial import cKDTree as KDTree

def cluster_MV(input,radius,acceptable_points):
    All= input
    rS=radius
    Mdl=KDTree(All)
    idx_S=Mdl.query_ball_point(All,rS)
    cluster=[]
    C=0
    for i in range(All[:,0]):
        L=len(idx_S[i,0])
        pp=np.argwhere([All[idx_S[i,0][1:L],3]>=1.0]) # ASK what???
        if pp == np.empty:
            C=C+1
            for j in range(L):
                All[idx_S[i,1][j],4]=C # Ask: What?
        else:
            mm=np.min(All[idx_S[i,1][pp],4])
            ma=np.max(All[idx_S[i,1][pp],4])
            for j in range(L):
                All[idx_S[i,1][j],4]=mm

            if ma!=mm:
                t=np.argwhere(All[:,4]==ma)
                All(t,4)=mm
