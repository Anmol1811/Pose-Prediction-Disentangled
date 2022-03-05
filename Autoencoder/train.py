import os
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
 
import numpy as np
import pickle
import glob
import time

import cv2
from Autoencoder import Autoencoder 
from utils import ADE_c, PoseDataset

def train(net, trainloader,valloader, NUM_EPOCHS, scheduler):
    train_loss = []
    val_losses = []
    max_val_loss=10000
    for epoch in range(NUM_EPOCHS):
        counter=0
        ade_train=0
        ade_val=0
        
        net.train()
        running_loss = 0.0
        running_val_loss = 0.0
        for data in trainloader:
            counter+=1
            kp = data['keypoints']
            kp = kp.to(device)
            
            optimizer.zero_grad()
            outputs = net(kp.float())
            
            loss = criterion(outputs, kp.float())
            
            ade_train += float(ADE_c(outputs.unsqueeze(0), kp.unsqueeze(0)))
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        
        ade_train/=counter
        
        net.eval()
        counter=0
        with torch.no_grad():
          for data in valloader:
              counter+=1
              kp = data['keypoints']
              kp = kp.to(device)
              outputs = net(kp.float())
              
              ade_val += float(ADE_c(outputs.unsqueeze(0), kp.unsqueeze(0)))
              
              val_loss = criterion(outputs, kp.float())
              running_val_loss += val_loss.item()
          
        val_loss = running_val_loss / len(valloader)
        val_losses.append(val_loss)
        ade_val/=counter
         

        scheduler.step(val_loss) 

#         if(epoch%20==19):
        print('Epoch {} of {}, Train Loss: {:.3f}, Val Loss: {:.3f}, ADE_train : {:.3f},  ADE_val: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss, val_loss,ade_train, ade_val))
              
        if val_loss<max_val_loss:
          max_val_loss=val_loss
          torch.save(net.state_dict(), "model_clean_auto.weight")

    return train_loss,val_losses

        


kpa=[]
keypoints=pickle.load(open("keypoints_openpifpaf.pickle","rb"))
kpd={}
for i in keypoints:
	kpd[i]=np.array(keypoints[i])
print(len(kpd))
kpa=[]
for key in list(kpd.keys()):
  x=key.replace(".png.predictions.json","").split("_")
  vid_num=int(x[2])
  frame_num=int(x[3])
  ped_num=x[4]
  if(len(ped_num)==11):
    ped_num=int(x[4][10])
  else:
    ped_num=0
    
  vec=kpd[key][2::3]
  if (vec > 0.25).all():
      kpa.append([vid_num,frame_num,ped_num,kpd[key]])


keypoints_array_sorted=sorted(kpa,key=lambda e:(e[0],e[2],e[1]))
print(len(keypoints_array_sorted))
thresarr=[]
for i in range(len(keypoints_array_sorted)):
  vec=keypoints_array_sorted[i][3][2::3]
  if (vec > 0.25).all():
      thresarr.append(keypoints_array_sorted[i][3])
keypoints=np.array(thresarr)
kp_train=np.delete(keypoints, list(range(2, keypoints.shape[1], 3)), axis=1)

np.random.shuffle(kp_train)
train_dataset=PoseDataset(kp_train[:int(len(kp_train)*0.7)])
val_dataset=PoseDataset(kp_train[:int(len(kp_train)*0.7)])
trainloader=DataLoader(train_dataset, batch_size=BATCH_SIZE)
valloader=DataLoader(val_dataset, batch_size=BATCH_SIZE)

net = Autoencoder()
device = torch.device('cuda:0')
net.to(device)
net = net.float()
NUM_EPOCHS = 300
BATCH_SIZE = 30
# net.load_state_dict(torch.load('model_best.weight'))

criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,verbose=True, factor=0.5)
train_loss, val_loss = train(net, trainloader, valloader, NUM_EPOCHS, scheduler)
