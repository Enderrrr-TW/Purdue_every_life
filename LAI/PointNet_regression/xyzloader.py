'''
Author: An-Te Huang
Time: 2021/12/25
'''
import os
import numpy as np
import warnings
import pickle
import json
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
def decide_npoints(data_dict):
    npoint_list=[]
    assert type(data_dict)==dict
    for i in data_dict['train']['path']:
        xyz=np.loadtxt(i)
        npoint_list.append(xyz.shape[0])
    for i in data_dict['test']['path']:
        xyz=np.loadtxt(i)
        npoint_list.append(xyz.shape[0])
    print('minimum number of points:',np.min(npoint_list))
    return np.min(npoint_list)

def randomly_sample(xyz,npoints):
    '''
    xyz: preprocessed point cloud
    npoints: output from decide_npoints
    return a sampled point cloud in order to build a tensor (all point clouds must have the same number of points)
    '''
    random_indice=np.random.choice(xyz.shape[0],size=npoints,replace=False)
    xyz_sampled=xyz[random_indice]
    return xyz_sampled

class LASDataLoader(Dataset):
    def __init__(self,root,split='train', preprocessed=True):
        super().__init__()
        self.root = '/home/tirgan/a/huan1577/dataset/testing'
        self.preprocessed = preprocessed
        self.split=split

        with open(self.root+'/data_dict.json','r') as json_file:
            self.data_dict=json.load(json_file)

        self.npoints = decide_npoints(self.data_dict)
        
        assert (split == 'train' or split == 'test')
        self.datasize=len(self.data_dict[split]['LAI'])
    def __len__(self):
        return self.datasize

    def _get_item(self,index):
        if self.preprocessed==True:
            xyz=np.loadtxt(self.data_dict[self.split]['path'][index])
            LAI=np.float32(self.data_dict[self.split]['LAI'][index])
            xyz_sampled=randomly_sample(xyz,self.npoints)
        else: 
            print('write this part by yourself dude')
            quit()
        return xyz_sampled,LAI
    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = LASDataLoader('/home/tirgan/a/huan1577/dataset/testing', split='test')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)
    for pts, LAI in DataLoader:
        print(pts.shape)#(batchsize,number of points, xyz+otherfeatures)
        print(LAI.shape)
