from utils import data_loader,draw_keypoints
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
        self.output = 16
        self.stride = 16
        self.skip   = 1
        self.lr = 0.01
        
args = args()
net = LSTM_vel(args).to(args.device)

train_loader=data_loader(args,"train",'sequences_openpifpaf_thres_9_')
val_loader=data_loader(args,"val" ,'sequences_openpifpaf_thres_9_')

for idx, (obs_s, target_s, obs_pose, target_pose, obs_mask, target_mask) in enumerate(val_loader):
        counter+=1
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_pose    = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        obs_mask    = obs_mask.to(device='cuda')
        target_mask = target_mask.to(device='cuda')
        
        (speed_preds,mask_preds) = net(pose=obs_pose,vel=obs_s,mask=obs_mask)
        preds_p = speed2pos(speed_preds, obs_pose)

        for b in range(args.batch_size)
            for i in range(args.output):
                actual=target_pose[b][i].detach().cpu().numpy()
                pred=preds_p[b][i].detach().cpu().numpy()
                
                x1=draw_keypoints(actual.astype(int))
                x2=draw_keypoints(pred.astype(int))
                
                f, axarr = plt.subplots(1,2)
                axarr[0].imshow(x1)
                axarr[1].imshow(x2)
                plt.show()

        