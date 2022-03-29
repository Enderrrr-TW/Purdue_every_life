# import modules
import pandas as pd
import numpy as np
import os

import json
from tqdm import tqdm

def train_test_split():
    test_plot_ID=[4360,4361,4380,4381,4400,4401,4420,4421,4440]
    input_folder='E:/Ender/data/testing/normalized'
    LAI=pd.read_csv('E:/Ender/data/testing/LAI_LiDAR_date.csv',index_col='plot')
    dataset={} 
    dataset['train']=dict()
    dataset['test']={}
    dataset['train']['path']=[]
    dataset['train']['LAI']=[]
    dataset['test']['path']=[]
    dataset['test']['LAI']=[]
    dates=os.listdir(input_folder)
    for date in dates:
        sub_dir=input_folder+'/'+date
        flist=os.listdir(sub_dir)
        for f in flist:
            fpath=sub_dir+'/'+f
            plot_ID=int(f[9:13])
            if plot_ID in test_plot_ID:
                dataset['test']['path'].append(fpath)
                dataset['test']['LAI'].append(LAI.loc[plot_ID,str(date)])
            else:
                dataset['train']['path'].append(fpath)
                dataset['train']['LAI'].append(LAI.loc[plot_ID,str(date)])
    # val_plot_ID=[4359,4362,4379,4382,4399,4402,4419,4422,4439]
    # train_plot_ID=[i for i in range(4351,4441)]
    # train_plot_ID=list(set(train_plot_ID)-set(val_plot_ID)-set(test_plot_ID))
    with open('E:/Ender/data/testing/train_test_split.json','w') as output:
        json.dump(dataset,output)
if __name__=='__main__':
    train_test_split()