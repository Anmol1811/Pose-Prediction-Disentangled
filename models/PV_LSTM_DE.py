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


class PV_LSTM_DE(nn.Module):
    def __init__(self, args):
        '''
           input: observed body poses and velocites global and local
           output: global and local velocities
        '''
        super(PV_LSTM_DE, self).__init__()
        self.in_s=34
        self.pose_encoder = nn.LSTM(input_size=self.in_s, hidden_size=args.hidden_size)
        self.vel_encoder = nn.LSTM(input_size=self.in_s, hidden_size=args.hidden_size)
        self.vel_decoder = nn.LSTMCell(input_size=self.in_s, hidden_size=args.hidden_size)
        self.fc_vel    = nn.Linear(in_features=args.hidden_size, out_features=self.in_s)
        
        self.pose_glob_encoder = nn.LSTM(input_size=2, hidden_size=100)
        self.vel_glob_encoder = nn.LSTM(input_size=2, hidden_size=100)
        self.vel_glob_decoder = nn.LSTMCell(input_size=2, hidden_size=100)
        self.fc_global    = nn.Linear(in_features=100, out_features=2)
        
        self.fc_combine= nn.Linear(in_features=self.in_s, out_features=self.in_s)
        
        self.hardtanh = nn.Hardtanh(min_val=-1*args.hardtanh_limit,max_val=args.hardtanh_limit)
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        
        self.args = args
        
    def forward(self, pose_local=None, vel_local=None, pose_glob=None, vel_glob=None):


        _, (hidden_vel, cell_vel) = self.vel_encoder(vel_local.permute(1,0,2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)


        _, (hidden_pose, cell_pose) = self.pose_encoder(pose_local.permute(1,0,2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)
        
        _, (hidden_vel_glob, cell_vel_glob) = self.vel_glob_encoder(vel_glob.permute(1,0,2))
        hidden_vel_glob = hidden_vel_glob.squeeze(0)
        cell_vel_glob = cell_vel_glob.squeeze(0)
        
        _, (hidden_pose_glob, cell_pose_glob) = self.pose_glob_encoder(pose_glob.permute(1,0,2))
        hidden_pose_glob = hidden_pose_glob.squeeze(0)
        cell_pose_glob = cell_pose_glob.squeeze(0)
        
        outputs = []
        
 
        vel_local_outputs    = torch.tensor([], device=self.args.device)
        vel_glob_outputs    = torch.tensor([], device=self.args.device)
        vel_combine_outputs = torch.tensor([], device=self.args.device)

        VelDec_inp = vel_local[:,-1,:]
        VelDec_glob_inp = vel_glob[:,-1,:]
        
        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        
        hidden_glob = hidden_pose_glob + hidden_vel_glob
        cell_glob = cell_pose_glob + cell_vel_glob
    
        
        for i in range(self.args.output//self.args.skip):
            hidden_dec, cell_dec = self.vel_decoder(VelDec_inp, (hidden_dec, cell_dec))
            vel_local_output  = self.hardtanh(self.fc_vel(hidden_dec))
            vel_local_outputs = torch.cat((vel_local_outputs, vel_local_output.unsqueeze(1)), dim = 1)
            VelDec_inp  = vel_local_output.detach()
            
            hidden_glob, cell_glob = self.vel_glob_decoder(VelDec_glob_inp, (hidden_glob, cell_glob))
            vel_glob_output  = self.hardtanh(self.fc_global(hidden_glob))
            vel_glob_outputs = torch.cat((vel_glob_outputs, vel_glob_output.unsqueeze(1)), dim = 1)
            VelDec_glob_inp  = vel_glob_output.detach()
            
            
            vel_local=vel_local_output.reshape(-1,self.in_s//2,2)
            vel_glob=vel_glob_output.reshape(-1,1,2)
            
            vel_combine=(vel_local+vel_glob).reshape(-1,self.in_s)
            
            vel_combine_output=self.fc_combine(vel_combine)
            
            vel_combine_outputs = torch.cat((vel_combine_outputs, vel_combine.unsqueeze(1)), dim = 1)#_output

            
        outputs.append(vel_local_outputs)
        outputs.append(vel_glob_outputs)
        outputs.append(vel_combine_outputs)
        
        return tuple(outputs)


class myDataset_DE_op(torch.utils.data.Dataset):
    def __init__(self, args, dtype):
        
        self.args = args
        self.dtype = dtype
        print("Loading",self.dtype)
        
        sequence_centric = pd.read_csv("sequences_16_overlap_4_thres4_"+self.dtype+".csv") 

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
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):

        seq = self.data.iloc[index]
        outputs = []

        obs = torch.tensor([seq.Pose[i] for i in range(0,self.args.input,self.args.skip)])        
        obs_speed = (obs[1:] - obs[:-1])
        
        true = torch.tensor([seq.Future_Pose[i] for i in range(0,self.args.output,self.args.skip)])
        true_speed = torch.cat(((true[0]-obs[-1]).unsqueeze(0), true[1:]-true[:-1]))
        

        obs_resh = torch.reshape(obs, (obs.size()[0],25,2))
        
        obs_global=obs_resh[:,1]
        obs_global=obs_global.unsqueeze(1)

        obs_resh=obs_resh-obs_global
    
        obs_resh=obs_resh.reshape(obs.size())
        
        true_resh = torch.reshape(true, (true.size()[0],25,2))
        true_global=true_resh[:,1]
        true_global=true_global.unsqueeze(1)
#         print(obs_resh.size(),obs_global.size())
        true_resh=true_resh-true_global
        true_resh=true_resh.reshape(true.size())
        
        
        obs_global=torch.reshape(obs_global, (obs.size()[0],2))
        true_global=torch.reshape(true_global, (true.size()[0],2))
        
        obs_local_speed=(obs_resh[1:] - obs_resh[:-1])
        true_local_speed=torch.cat(((true_resh[0]-obs_resh[-1]).unsqueeze(0), true_resh[1:]-true_resh[:-1]))
        
        obs_global_speed = (obs_global[1:] - obs_global[:-1])
        true_global_speed = torch.cat(((true_global[0]-obs_global[-1]).unsqueeze(0), true_global[1:]-true_global[:-1]))
        
#         print(obs_resh.size(),true_resh.size(),obs_global.size(),true_global.size())
        
        outputs.append(obs)
        outputs.append(obs_resh)
        outputs.append(true_resh)
        outputs.append(obs_global)
        outputs.append(true_global)
        outputs.append(obs_local_speed)
        outputs.append(true_local_speed)
        outputs.append(obs_global_speed)
        outputs.append(true_global_speed)
        outputs.append(true_speed)
        outputs.append(true)

        
        return tuple(outputs)    
    
    
def data_loader_DE_op(args,data):
    dataset = myDataset_DE_op(args,data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory)

    return dataloader


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

train_loader=data_loader_DE_op(args,"train",)
val_loader=data_loader_DE_op(args,"val")


optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, 
                                                 threshold = 1e-8, verbose=True)
print('='*100)
print('Training ...')
alpha=0.3

for epoch in range(200):
    start = time.time()
    
    avg_epoch_train_s_loss = 0
    avg_epoch_val_s_loss   = 0
    avg_epoch_train_p_loss   = 0
    avg_epoch_val_p_loss     = 0 
    avg_epoch_train_g_loss   = 0
    avg_epoch_val_g_loss     = 0
    avg_epoch_train_loss   = 0
    avg_epoch_val_loss     = 0 
    ade  = 0
    fde  = 0
    ade_train  = 0
    fde_train  = 0
    counter = 0
    
    for idx, (obs_pose, obs_pose_local, target_pose_local, obs_pose_global, target_pose_global, obs_s_l, target_s_l, obs_s_g, target_s_g, target_s, target) in enumerate(train_loader):
        counter += 1        
               
        obs_pose = obs_pose.to(device='cuda')
        obs_pose_local    = obs_pose_local.to(device='cuda')
        target_pose_local = target_pose_local.to(device='cuda')
        obs_pose_global = obs_pose_global.to(device='cuda')
        target_pose_global = target_pose_global.to(device='cuda')
        obs_s_l  = obs_s_l.to(device='cuda')
        target_s_l = target_s_l.to(device='cuda')
        obs_s_g  = obs_s_g.to(device='cuda')
        target_s_g = target_s_g.to(device='cuda')
        target_s = target_s.to(device='cuda')
        target = target.to(device='cuda')
        
        net.zero_grad()
    
        (speed_preds,global_preds,target_preds) = net(pose_local=obs_pose_local, vel_local=obs_s_l, pose_glob=obs_pose_global, vel_glob=obs_s_g)

        speed_loss  = l1e(speed_preds, target_s_l)
        global_loss =  l1e(global_preds, target_s_g)
        act_loss = l1e(target_preds,target_s)
        
        loss = 0.25*speed_loss + 0.25*global_loss + 0.5*act_loss
        loss.backward()
    
        preds_p=speed2pos(target_preds,obs_pose)
        
        ade_train += float(ADE_c(preds_p, target))
        fde_train += float(FDE_c(preds_p, target))
        
        optimizer.step()
        
        avg_epoch_train_s_loss += float(speed_loss)
        avg_epoch_train_g_loss += float(global_loss)
        avg_epoch_train_loss += float(act_loss)
        
    avg_epoch_train_s_loss /= counter
    avg_epoch_train_g_loss /= counter
    avg_epoch_train_loss /= counter
    
    ade_train  /= counter
    fde_train  /= counter  
    counter=0

    for idx, (obs_pose, obs_pose_local, target_pose_local, obs_pose_global, target_pose_global, obs_s_l, target_s_l, obs_s_g, target_s_g, target_s, target) in enumerate(val_loader):
        counter+=1
        
        obs_pose = obs_pose.to(device='cuda')
        obs_pose_local = obs_pose_local.to(device='cuda')
        target_pose_local = target_pose_local.to(device='cuda')
        obs_pose_global = obs_pose_global.to(device='cuda')
        target_pose_global = target_pose_global.to(device='cuda')
        obs_s_l  = obs_s_l.to(device='cuda')
        target_s_l = target_s_l.to(device='cuda')
        obs_s_g  = obs_s_g.to(device='cuda')
        target_s_g = target_s_g.to(device='cuda')
        target_s = target_s.to(device='cuda')
        target = target.to(device='cuda')
        

        
        with torch.no_grad():
            
            (speed_preds,global_preds,target_preds) = net(pose_local=obs_pose_local, vel_local=obs_s_l, pose_glob=obs_pose_global, vel_glob=obs_s_g)
            
            speed_loss  = l1e(speed_preds, target_s_l)
            global_loss =  l1e(global_preds, target_s_g)
            act_loss = l1e(target_preds,target_s)

            val_loss = 0.25*speed_loss + 0.25*global_loss + 0.5*act_loss
            
            preds_p=speed2pos(target_preds,obs_pose)
            
            avg_epoch_val_s_loss += float(speed_loss)
            avg_epoch_val_g_loss += float(global_loss)
            avg_epoch_val_loss += float(act_loss)
            
            ade += float(ADE_c(preds_p, target))
            fde += float(FDE_c(preds_p, target))

        
    avg_epoch_val_s_loss /= counter
    avg_epoch_val_g_loss /= counter
    avg_epoch_val_loss /= counter
    
    ade  /= counter
    fde  /= counter     
    
    scheduler.step(val_loss)
    
     
    print('e:', epoch, '| ts: %.2f'% avg_epoch_train_s_loss, '| tg: %.2f'% avg_epoch_train_g_loss, '| vs: %.2f'% avg_epoch_val_s_loss, '| vg: %.2f'% avg_epoch_val_g_loss, '| t: %.2f'% avg_epoch_train_loss, '| v: %.2f'% avg_epoch_val_loss, '| ade_train: %.2f'% ade_train, '| ade_val: %.2f'% ade, '| fde_train: %.2f'% fde_train,'| fde_val: %.2f'% fde)



print('='*100) 
print('Done !')