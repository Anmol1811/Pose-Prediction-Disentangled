import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.drop =  nn.Dropout(p=0.3)
        self.enc1= nn.Linear(in_features=34, out_features=30)
        self.enc2 = nn.Linear(in_features=30, out_features=15)
        # decoder 
        self.dec1 = nn.Linear(in_features=15, out_features=30)
        self.dec2 = nn.Linear(in_features=30, out_features=34)
        
    def forward(self, x):
                
        x = F.relu(self.drop(x))
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x
        
net = Autoencoder()
print(net)