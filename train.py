from utils import data_loader
from utils import ADE_c, FDE_c, speed2pos
from models.LSTM_vel import LSTM_vel


import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
 
from torchvision import datasets
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
import glob

from PIL import Image, ImageDraw
import time

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import cv2




class args():
    def __init__(self):
       
        self.loader_workers = 1
        self.loader_shuffle = False
        self.pin_memory     = False
        self.device         = 'cuda'
        self.batch_size     = 50
        self.n_epochs       = 200
        self.hidden_size    = 1000
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 14
        self.stride = 16
        self.skip   = 1
        self.lr = 0.01
        
args = args()
net = LSTM_vel(args).to(args.device)

optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
                                                 threshold = 1e-8, verbose=True)
mse = nn.MSELoss()

l1e = nn.L1Loss()
bce = nn.BCELoss()
train_s_scores = []
train_pose_scores=[]
val_pose_scores=[]
train_c_scores = []
val_s_scores   = []
val_c_scores   = []

train_loader=data_loader(args,"train",'sequences_openpifpaf_thres_9_')
val_loader=data_loader(args,"val" ,'sequences_openpifpaf_thres_9_')


optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, 
                                                 threshold = 1e-8, verbose=True)
print('='*100)
print('Training ...')

for epoch in range(args.n_epochs):
    start = time.time()
    
    avg_epoch_train_s_loss = 0
    avg_epoch_val_s_loss   = 0
    avg_epoch_train_p_loss   = 0
    avg_epoch_val_p_loss     = 0 
    ade  = 0
    fde  = 0
    ade_train  = 0
    fde_train  = 0
    counter = 0
    
    for idx, (obs_s, target_s, obs_pose, target_pose) in enumerate(train_loader):
        counter += 1        
        
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_pose    = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
 
        net.zero_grad()
    
        (speed_preds,) = net(pose=obs_pose, vel=obs_s)

        speed_loss  = l1e(speed_preds, target_s)
    
        preds_p = speed2pos(speed_preds, obs_pose) 
        ade_train += float(ADE_c(preds_p, target_pose))
        fde_train += float(FDE_c(preds_p, target_pose))
        
    
        speed_loss.backward()

        optimizer.step()
        
    
        avg_epoch_train_s_loss += float(speed_loss)

    avg_epoch_train_s_loss /= counter
    train_s_scores.append(avg_epoch_train_s_loss)
    ade_train  /= counter
    fde_train  /= counter    

    counter=0

    for idx, (obs_s, target_s, obs_pose, target_pose) in enumerate(val_loader):
        counter+=1
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_pose    = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        
        with torch.no_grad():
            (speed_preds,) = net(pose=obs_pose,vel=obs_s)

            speed_loss  = l1e(speed_preds, target_s)
            avg_epoch_val_s_loss += float(speed_loss)
        
            preds_p = speed2pos(speed_preds, obs_pose)
            ade += float(ADE_c(preds_p, target_pose))
            fde += float(FDE_c(preds_p, target_pose))

        
    avg_epoch_val_s_loss /= counter
    val_s_scores.append(avg_epoch_val_s_loss)
    
    ade  /= counter
    fde  /= counter     
   
    scheduler.step(avg_epoch_val_s_loss)
    
     
    print('e:', epoch, '| ts: %.2f'% avg_epoch_train_s_loss,  '| vs: %.2f'% avg_epoch_val_s_loss, '| ade_train: %.2f'% ade_train, '| ade_val: %.2f'% ade, '| fde_train: %.2f'% fde_train,'| fde_val: %.2f'% fde,
          '| t:%.2f'%(time.time()-start))


print('='*100) 
# print('Saving ...')
# torch.save(net.state_dict(), args.model_path)
print('Done !')