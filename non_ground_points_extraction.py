import CSF 
import numpy as np
import laspy
from laspy.file import File

import pandas as pd
import os

'''
Read the code and check whether it extracts non ground points or ground points
'''
def DEM(inFile,output_name):
    points = inFile.points
    # use negative z for DEM
    #------------------------testing--------------------
    h=np.quantile(inFile.z,0.995)
    idx=inFile.z<h+0.5
    print(h)
    # points=points[idx]
    #---------------------------------------------------
    xyz = np.vstack((inFile.x[idx], inFile.y[idx], inFile.z[idx])).transpose() # extract x, y, z and put into a list
    csf = CSF.CSF()
    # prameter settings
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = 1 # default=2
    csf.params.class_threshold = 0.1 # buffer for filter
    csf.params.rigidness = 3 # cloth softness, smaller softer
    ################################################
    #### csf.params.time_step = 0.65;
    #### csf.params.interations = 500;
    #### more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/
    ################################################

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
    non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
    csf.do_filtering(ground, non_ground) # do actual filtering.
    print(type(ground))
    outFile = File(output_name,mode='w', header=inFile.header)
    # outFile.points = points[ground] # extract ground points, and save it to a las file.
    outFile.points = points[non_ground] # extract ground points, and save it to a las file.

    outFile.close() # do not forget this
def main():
    input_folder='E:/Ender/data/backpack/raw/'
    output_folder='E:/Ender/data/backpack/non_ground/'
    temmp=os.listdir(input_folder)
    flist=[]
    for i in temmp:
        if i.endswith('.las'):
            output_name=output_folder+i.replace('.las','_ng.las')
            flist.append(i)
            fpath=input_folder+i
            las= File(fpath, mode='r')
            DEM(las,output_name)

if __name__ == '__main__':
    main()
