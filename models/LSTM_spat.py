import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms

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

from utils import data_loader
from utils import ADE_c, FDE_c, speed2pos

class args():
    def __init__(self):
        self.loader_workers = 1
        self.loader_shuffle = True
        self.pin_memory     = False
        self.device         = 'cuda'
        self.batch_size     = 50
        self.n_epochs       = 1000
        self.hidden_size    = 1000
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 16
        self.stride = 16
        self.skip   = 1
        self.lr = 0.01
args = args()


mse = nn.MSELoss()
l1e = nn.L1Loss()
train_s_scores = []
train_pose_scores=[]
val_pose_scores=[]
train_c_scores = []
val_s_scores   = []
val_c_scores   = []

train_loader=data_loader(args,"train","sequences_openpifpaf_")
val_loader=data_loader(args,"val","sequences_openpifpaf_")

class LSTM_spat(nn.Module):
    def __init__(self, args):
     
        super(LSTM_spat, self).__init__()


        self.encoded_size=20
         
        self.pose_encoder = nn.LSTM(input_size=self.encoded_size, hidden_size=args.hidden_size)        
        self.pose_embedding = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=self.encoded_size),
                                           nn.ReLU())
        
        self.pose_decoder = nn.LSTMCell(input_size=self.encoded_size, hidden_size=args.hidden_size)
        self.fc_pose   = nn.Linear(in_features=args.hidden_size, out_features=self.encoded_size)
        
        self.hardtanh = nn.Hardtanh(min_val=-1*args.hardtanh_limit,max_val=args.hardtanh_limit)
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.enc1= nn.Linear(in_features=34, out_features=30)
        self.enc2 = nn.Linear(in_features=30, out_features=self.encoded_size)
        self.dec = nn.Linear(in_features=self.encoded_size, out_features=34)
        

        
        self.args = args
        
    def forward(self, pose=None, vel=None, target_pose=None):
        
        poses=pose.permute(1,0,2)

        pose_encoded = torch.tensor([], device=self.args.device)
        
        
        for i in range(pose.size()[0]):
            x = self.relu(self.enc1(pose[i]))
            x = self.relu(self.enc2(x))
            recreated_pose=self.relu(self.dec(x))
            pose_encoded = torch.cat((pose_encoded, x.unsqueeze(1)), dim = 1)
        
        
        _, (hidden_pose, cell_pose) = self.pose_encoder(pose_encoded)
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)
        
        outputs = []
        pose_outputs   = torch.tensor([], device=self.args.device)
        PoseDec_inp = pose_encoded.permute(1,0,2)[:,-1,:]
        

        hidden_dec=hidden_pose
        cell_dec=cell_pose

        for i in range(self.args.output//self.args.skip):
        
            hidden_dec, cell_dec = self.pose_decoder(PoseDec_inp, (hidden_dec, cell_dec))
            pose_output_encoded  = self.fc_pose(hidden_dec)
            
            pose_output= self.relu(self.dec(pose_output_encoded))
            pose_outputs = torch.cat((pose_outputs, pose_output.unsqueeze(1)), dim = 1)
            PoseDec_inp  = pose_output_encoded.detach()
            
        
        pose_recs= pose_recs.permute(1,0,2)
        

        outputs.append(pose_outputs)
        outputs.append(pose_recs)

            
        return tuple(outputs)

net = LSTM_spat(args).to(args.device)


optimizer = optim.Adam(net.parameters(), lr= 0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, 
                                                 threshold = 1e-12, verbose=True)
print('='*100)
print('Training ...')

train_p_scores=[]
val_p_scores=[]

for epoch in range(200):
    start = time.time()
    
    avg_epoch_train_p_loss   = 0
    avg_epoch_val_p_loss     = 0 
    ade  = 0
    fde  = 0
    ade_train  = 0
    fde_train  = 0
    counter = 0
    net.train()
    for idx, (obs_s, target_s, obs_pose, target_pose) in enumerate(train_loader):
        counter += 1        
        
        
        obs_pose    = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
 
        net.zero_grad()
    
        (pose_preds,) = net(pose=obs_pose, target_pose=target_pose,vel=obs_s)

        loss  = l1e(pose_preds, target_pose)
    
        ade_train += float(ADE_c(pose_preds, target_pose))
        fde_train += float(FDE_c(pose_preds, target_pose))
        
    
        loss.backward()
        optimizer.step()
        
    
        avg_epoch_train_p_loss += float(loss)

    avg_epoch_train_p_loss /= counter
    train_p_scores.append(avg_epoch_train_p_loss)
    ade_train  /= counter
    fde_train  /= counter   
    
    
  
    counter=0
    net.eval()
    for idx, (obs_s, target_s, obs_pose, target_pose) in enumerate(val_loader):
        counter+=1
       
        obs_pose    = obs_pose.to(device='cuda')
        target_pose = target_pose.to(device='cuda')
        
        with torch.no_grad():
            (pose_preds,) = net(pose=obs_pose, target_pose=target_pose)
          
            val_loss  = l1e(pose_preds, target_pose)
            avg_epoch_val_p_loss += float(val_loss)
            
            ade += float(ADE_c(pose_preds, target_pose))
            fde += float(FDE_c(pose_preds, target_pose))

        
    avg_epoch_val_p_loss /= counter
    val_p_scores.append(avg_epoch_val_p_loss)
    
    ade  /= counter
    fde  /= counter     
   
    
    
    scheduler.step(avg_epoch_train_p_loss)
    
    print('e:', epoch,'| loss: %.2f'%avg_epoch_train_p_loss,'| val_loss: %.2f'% avg_epoch_val_p_loss, '| tpose: %.2f'% avg_epoch_train_p_loss, '| vpose: %.2f'% avg_epoch_val_p_loss, '| ade_train: %.2f'% ade_train, '| ade_val: %.2f'% ade, '| fde_train: %.2f'% fde_train,'| fde_val: %.2f'% fde,
          '| t:%.2f'%(time.time()-start))


print('='*100) 

print('Done !')