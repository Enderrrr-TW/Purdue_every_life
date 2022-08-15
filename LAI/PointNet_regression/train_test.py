"""
Origin
Author: Benny
Date: Nov 2019
---
Revised by An-Te
Date: Dec 2021
Purpose: need a version for regression
"""

import os
import sys
from numpy.core.defchararray import mod
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from data_utils.xyzloader import LASDataLoader_kfold as LASDataLoader

### Fix random seed
torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)
###
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
torch.cuda.empty_cache()# release memory

# def validation_set_selector(root,k):
#     with open(root+'/'+f'train_test_split_fold{k}.json','r') as json_file:
#         data_dict=json.load(json_file)
#     date_list=[]
#     for i in range(len(data_dict['train']['path'])):
#         rlist=re.split('/',data_dict['train']['path'][i])
#         date_list.append(rlist[-2])
#     skf=StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
#     skf.get_n_splits(data_dict['train']['path'],date_list)
#     for train_index, valid_index in skf.split(data_dict['train']['path'],date_list):
#         return train_index,valid_index
def validation_set_selector(root,k):
    with open(root+'/'+f'train_test_split_fold{k}.json','r') as json_file:
        data_dict=json.load(json_file)
    stratified_feature=[]
    for i in range(len(data_dict['train']['path'])):
        rlist=re.split('/',data_dict['train']['path'][i])
        plot_ID=int(re.split('_',data_dict['train']['path'][i])[6])
        if plot_ID<4395:
            hi_temp=1
        else:
            hi_temp=0
        stratified_feature.append(rlist[-2]+str(hi_temp))

    skf=StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
    skf.get_n_splits(data_dict['train']['path'],stratified_feature)
    for train_index, valid_index in skf.split(data_dict['train']['path'],stratified_feature):
        return train_index,valid_index

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_reg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--fold',type=int, help='index of fold')
    parser.add_argument('--data', type=str,help='HIPS_2020 or HIPS_2021')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    return parser.parse_args()
def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/UAV/'
    # train_dataset = LASDataLoader(root=data_path, split='train', preprocessed=True, sampled=True)

    train_dataset = LASDataLoader(data=args.data,fold=args.fold, split='train', preprocessed=True,sampled=True)
    # test_dataset = LASDataLoader(root=data_path, split='test', preprocessed=True)
    print('train data size:',train_dataset.datasize)
    '''MODEL LOADING'''
    model = importlib.import_module(args.model)

    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_regression.py', str(exp_dir))
    k=1
    classifier = model.get_model(k)
    criterion = model.get_loss()
    ###################### parallel########################
    # device = torch.device("cuda")
    # classifier=torch.nn.DataParallel(classifier, device_ids=[0,1])
    # classifier.to(device)
    # criterion=torch.nn.DataParallel(criterion, device_ids=[0,1])
    # criterion.to(device)
    #######################################################
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        training_loss_record=np.loadtxt('/home/tirgan/a/huan1577/Pointnet_Pointnet2_pytorch/log/classification/pointnet_reg/training_loss.npy')
        training_loss_record=list(training_loss_record)
        R2_record=np.loadtxt('/home/tirgan/a/huan1577/Pointnet_Pointnet2_pytorch/log/classification/pointnet_reg/training_loss.npy')
        R2_record=list(R2_record)
        best_loss=np.min(training_loss_record)
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        training_loss_record=[] # 2D: record all training loss in all folds
        R2_record=[]#1D: only record the corresponding R2
        best_loss=878787878787
        validation_loss_record=[]
    
    train_ids,valid_ids=validation_set_selector(train_dataset.root, args.fold) 
    # classifier.apply(reset_weights)
    start_epoch = 0
    best_loss=878787878787
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10,sampler=train_subsampler)
    validDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10,sampler=valid_subsampler)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0


    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        # if epoch==100:
        #     optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        total_training_loss=0
        m=0
        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target, trans_feat)
            # loss.sum().backward()
            loss.backward()
            training_loss_record.append(loss.detach().to('cpu').numpy())
            optimizer.step()
            global_step += 1
            total_training_loss=total_training_loss+loss*len(target)
            m=m+len(target)
            
        # log_string('Train loss: %f' % loss.sum())
        log_string('Train loss: %f' % loss)

        with torch.no_grad():
            total_training_loss=total_training_loss/m
            res=[]
            total_loss=0
            classifier.eval()
            n=0
            for j, (points, targetv) in tqdm(enumerate(validDataLoader), total=len(validDataLoader)):
                if not args.use_cpu:
                    points, targetv = points.cuda(), targetv.cuda()
                points = points.transpose(2, 1).to(torch.float)
                predv, _ = classifier(points)
                torch_valid_loss=criterion(predv,targetv,trans_feat)
                # total_loss=total_loss+torch_valid_loss.mean()*len(target)
                total_loss=total_loss+torch_valid_loss*len(targetv)

                n+=len(targetv)
            res=np.array(res).flatten()
            total_loss=total_loss/n
            validation_loss_record.append(total_loss.cpu().numpy())
            log_string('Validation loss: %f' % total_loss)
            # valid_R2=r2_score(target_all,pred_all)
            if (total_loss < best_loss):

                best_loss=total_loss
                corresponding_training_loss= total_training_loss
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + f'/best_model.pth'
                best_epoch = epoch + 1
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'loss' : best_loss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
        
            global_epoch += 1
    logger.info('End of training...')
    # print('training loss:',np.min(training_loss_record))
    # print('Corresponding R**2:',best_R2)
    log_string('Best Validation loss: %f' % best_loss)
    log_string('Best Epoch: %s' % best_epoch)
    log_string('Corresponding training error:%f' % corresponding_training_loss)
    np.savetxt('/home/tirgan/a/huan1577/Pointnet_Pointnet2_pytorch/log/classification/'+args.log_dir+'/training_loss.npy',np.array(training_loss_record))
    np.savetxt('/home/tirgan/a/huan1577/Pointnet_Pointnet2_pytorch/log/classification/'+args.log_dir+'/validation_loss.npy',np.array(validation_loss_record))
if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(42)
    main(args)
