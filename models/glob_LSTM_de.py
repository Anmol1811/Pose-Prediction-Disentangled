import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
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
import cv2



class myDataset_DE_depth(torch.utils.data.Dataset):
    def __init__(self, args, dtype):
        
        self.args = args
        self.dtype = dtype
        print("Loading",self.dtype)
        
        sequence_centric = pd.read_csv("glob_depth_"+self.dtype+".csv")

        df = sequence_centric.copy()      
        for v in list(df.columns.values):
            print(v+' loaded')
            try:
                df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
            except:
                continue
        sequence_centric[df.columns] = df[df.columns]
        self.data = sequence_centric.copy().reset_index(drop=True)
        
        print('*'*30)
        
        self.obs=self.data.Pose
        self.true=self.data.Future_Pose
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        
        outputs=[]
        
        obs=torch.tensor(self.obs[index],dtype=torch.double)
        true=torch.tensor(self.true[index],dtype=torch.double)
        
        outputs.append(obs[:,:2])
        outputs.append(true)
        obs_speed = (obs[1:,:2] - obs[:-1,:2])
        obs_speed=torch.cat((obs_speed,obs[1:,2:]),dim=1)
        
        outputs.append(obs_speed)
        
        true_speed = torch.cat(((true[0,:2]-obs[-1,:2]).unsqueeze(0), true[1:,:2]-true[:-1,:2]))
        outputs.append(true_speed)

       
        return tuple(outputs)    
    
    
def data_loader_DE_depth(args,data):
    dataset = myDataset_DE_depth(args,data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory)

    return dataloader


class glob_LSTM_DE(nn.Module):    
    def __init__(self, args):
    
        super(glob_LSTM_DE, self).__init__()
         
        
        self.DE_encoder = nn.LSTM(input_size=15, hidden_size=100)
        self.DE_decoder = nn.LSTMCell(input_size=15, hidden_size=100)

        self.fc_DE  = nn.Linear(in_features=100, out_features=15)
        self.fc2_DE  = nn.Linear(in_features=15, out_features=2)
        
        self.hardtanh = nn.Hardtanh(min_val=-1*100,max_val=100)
        self.relu = nn.ReLU() 
        
        self.args = args
        
    def forward(self, pose=None):
        
        outputs = []        
        _, (hidden_dec,cell_dec) = self.DE_encoder(pose.permute(1,0,2))    
        hidden_dec=hidden_dec.squeeze(0)
        cell_dec=cell_dec.squeeze(0)

        DEDec_inp = pose[:,-1,:]
       
        DE_outputs = torch.tensor([], device=self.args.device,dtype=torch.double)
        
        for i in range(self.args.output//self.args.skip):

            (hidden_dec,cell_dec) = self.DE_decoder(DEDec_inp, (hidden_dec,cell_dec)) #decoder_output, 
            DE_output_t  = self.fc_DE(hidden_dec)
            DE_output  = self.fc2_DE(DE_output_t)
            DE_outputs = torch.cat((DE_outputs, DE_output.unsqueeze(1)), dim = 1)
            DEDec_inp  = DE_output_t.detach()
            
            
        outputs.append(DE_outputs)
            
        return tuple(outputs)

net=glob_LSTM_DE(args).to(args.device).double()


class args():
    def __init__(self):
       
        self.dtype        = 'train'
        self.loader_workers = 1
        self.loader_shuffle = True
        self.pin_memory     = False
        self.device         = 'cuda'
        self.batch_size     = 50
        self.input  = 16
        self.output = 16
        self.stride = 16
        self.skip   = 1
        
args = args()

train_loader=data_loader_DE_depth(args,"train")
val_loader=data_loader_DE_depth(args,"val" )

print('='*100)
print('Training ...')

train_p_scores=[]
val_p_scores=[]
alpha=1#0.4
l1e = nn.L1Loss()
train_s_scores = []
train_pose_scores=[]
val_pose_scores=[]
train_c_scores = []
val_s_scores   = []
val_c_scores   = []

optimizer = optim.Adam(net.parameters(), lr= 0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, 
                                                 threshold = 1e-12, verbose=True)

for epoch in range(300):
    start = time.time()
    
    avg_epoch_train_p_loss   = 0
    avg_epoch_val_p_loss     = 0 
    ade_g  = 0
    fde_g  = 0
    ade_train_g  = 0
    fde_train_g  = 0
    counter = 0
    net.train()
    
    for idx, (obs_pose_global, target_pose_global,obs_s_global,target_s_global) in enumerate(train_loader):
        obs_pose_global = obs_pose_global.to(device='cuda')
        target_pose_global = target_pose_global.to(device='cuda')
        obs_s_global = obs_s_global.to(device='cuda')
        target_s_global = target_s_global.to(device='cuda')
        
        counter += 1        
        
        net.zero_grad()

        (preds,) = net(pose=obs_s_global)
        preds_g=speed2pos(preds,obs_pose_global)
        
        ade_train_g += float(ADE_c(preds_g, target_pose_global))
        fde_train_g += float(FDE_c(preds_g, target_pose_global))
      
        loss_g  = l1e(preds, target_s_global)
        
        loss=alpha*loss_g
    
        loss.backward()
        optimizer.step()
        
        avg_epoch_train_p_loss += float(loss)

    avg_epoch_train_p_loss /= counter
    ade_train_g  /= counter
    fde_train_g  /= counter   
    

    counter=0
    net.eval()
    for idx, (obs_pose_global, target_pose_global,obs_s_global,target_s_global) in enumerate(val_loader):
        obs_pose_global = obs_pose_global.to(device='cuda')
        target_pose_global = target_pose_global.to(device='cuda')
        obs_s_global = obs_s_global.to(device='cuda')
        target_s_global = target_s_global.to(device='cuda')
    
        counter += 1        

        with torch.no_grad():
            (preds,) = net(pose=obs_s_global)
            
            preds_g=speed2pos(preds,obs_pose_global)
        
            ade_g += float(ADE_c(preds_g, target_pose_global))
            fde_g += float(FDE_c(preds_g, target_pose_global))

            loss_g  = l1e(preds, target_s_global)

            val_loss=alpha*loss_g

            avg_epoch_val_p_loss += float(val_loss)
      
    avg_epoch_val_p_loss /= counter
    val_p_scores.append(avg_epoch_val_p_loss)
    ade_g  /= counter
    fde_g  /= counter    
   
    
    
    scheduler.step(avg_epoch_val_p_loss)
    
    print('e:', epoch,'| loss: %.2f'%avg_epoch_train_p_loss,'| val_loss: %.2f'% avg_epoch_val_p_loss, '| ade_train_g: %.2f'% ade_train_g, '| ade_val_g: %.2f'% ade_g, '| fde_train_g: %.2f'% fde_train_g,'| fde_val_g: %.2f'% fde_g)

print('='*100) 
# print('Saving ...')
# torch.save(net.state_dict(), args.model_path)
print('Done !')