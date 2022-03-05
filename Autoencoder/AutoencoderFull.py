import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoencoderFull(nn.Module):
    def __init__(self):
        super(AutoencoderFull, self).__init__()
        # encoder
        self.drop =  nn.Dropout(p=0.3)
        self.enc1= nn.Linear(in_features=34*16, out_features=300)
        self.enc2 = nn.Linear(in_features=300, out_features=150)
        self.enc3 = nn.Linear(in_features=150, out_features=75)
        # self.enc4 = nn.Linear(in_features=75, out_features=10)
        # decoder 
        # self.dec1 = nn.Linear(in_features=10, out_features=300)
        self.dec2 = nn.Linear(in_features=75, out_features=150)
        self.dec3 = nn.Linear(in_features=300, out_features=300)
        self.dec4 = nn.Linear(in_features=300, out_features=34*16)
        
    def forward(self, x):
                
        x = F.relu(self.drop(x))
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
#         x = F.relu(self.enc4(x))
#         x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        return x
net = AutoencoderFull()
print(net)