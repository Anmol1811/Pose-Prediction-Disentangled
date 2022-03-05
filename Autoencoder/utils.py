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

import cv2

def FDE_c(pred, true):
    b,n,p=pred.size()[0],pred.size()[1],pred.size()[2]
#     print(b,n,p)
    pred = torch.reshape(pred, (b,n,int(p/2),2))
    true = torch.reshape(true, (b,n,int(p/2),2))
    
    displacement=torch.sqrt((pred[:,-1,:,0]-true[:,-1,:,0])**2+(pred[:,-1,:,1]-true[:,-1,:,1])**2)

    fde = torch.mean(torch.mean(displacement,dim=1))
    
    return fde

def speed2pos(preds, obs_p):
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], preds.shape[2]).to('cuda')
    current = obs_p[:,-1,:]
    
    for i in range(preds.shape[1]):
        pred_pos[:,i,:] = current + preds[:,i,:]
        current = pred_pos[:,i,:]
        
    for i in range(preds.shape[2]):
        pred_pos[:,:,i] = torch.min(pred_pos[:,:,i], 1920*torch.ones(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        pred_pos[:,:,i] = torch.max(pred_pos[:,:,i], torch.zeros(pred_pos.shape[0], pred_pos.shape[1], device='cuda'))
        
    return pred_pos



class PoseDataset(Dataset):

    def __init__(self,keypoints, transform=None):
        self.data=keypoints
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]

        sample = {'keypoints': torch.from_numpy(image)}

        return sample
