"""
Author: Benny
Date: Nov 2019
Source:https://github.com/yanx27/Pointnet_Pointnet2_pytorch
---
Revised by An-Te Huang
Date: Dec 31 2021
Purpose: Create a script for regression
"""
from tkinter.tix import Tree
from parso import parse
from data_utils.xyzloader import LASDataLoader_kfold
import argparse
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import sys
import importlib
from sklearn.metrics import mean_squared_error as skMSE
from sklearn.metrics import r2_score

torch.backends.cudnn.deterministic = True
### Fix random seed
torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)
###
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--fold', type=int, default=0,help='index of fold')
    parser.add_argument('--data', type=str, help='HIPS_2020 or HIPS_2021')
    # parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    return parser.parse_args()


def test(model,loader):
    for m in model.modules():
        for child in m.children():
            if type(child) == torch.nn.BatchNorm1d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None
    classifier = model.eval()
    # classifier=model
    # classifier=model.train()
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        n=0
        torch_loss=0
        res=[]
        pred_all=[]
        target_all=[]
        for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            points = points.transpose(2, 1).to(torch.float)
            pred, _ = classifier(points)
            MSEloss=criterion(pred,target)
            torch_loss=torch_loss+MSEloss.to('cpu').numpy()*len(target)
            n+=len(target)
            pred2=pred.to('cpu').numpy()
            pred2=pred2.flatten()
    
            target2=target.to('cpu').numpy()
            for i in range(len(pred2)):
                pred_all.append(pred2[i])
                target_all.append(target2[i])
                res.append(pred2[i]-target2[i])
        
        res=np.array(res).flatten()
        loss=skMSE(target_all,pred_all)
        plt.scatter(target_all,pred_all)
        plt.savefig('scatter_train.png')
        R2=r2_score(target_all,pred_all)
        state_dict=classifier.state_dict()
        return loss, R2, state_dict


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'remember to change this'

    test_dataset = LASDataLoader_kfold(data=args.data,fold=args.fold, split='test', preprocessed=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = 1
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    classifier = model.get_model(num_class, normal_channel=False)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters())
    # print('total:',pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print('total trainable:',pytorch_total_params)
    
    if not args.use_cpu:
        classifier = classifier.cuda()
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model-fold-3.pth')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')

    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_loss,test_R2,test_state_dict = test(classifier.eval(), testDataLoader)
        log_string('Test mean squared error: %f' % (test_loss))

    return test_loss, test_R2

if __name__ == '__main__':
    args = parse_args()
    [loss,R2]=main(args)

