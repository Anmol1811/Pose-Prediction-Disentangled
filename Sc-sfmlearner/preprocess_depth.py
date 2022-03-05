import torch

from imageio import imread, imsave
from skimage.transform import resize as imresize
import matplotlib.pyplot as plt

import numpy as np

import argparse
from tqdm import tqdm
import os 

from models import DispResNet
from utils2 import tensor2array
import glob


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
img_height=256
img_width=832
img_exts=['png', 'jpg', 'bmp']

resnet_layers=18
output_dir='results/'
pretrained='checkpoints/resnet18_depth_256/dispnet_model_best.pth.tar'

with torch.no_grad():


    disp_net = DispResNet(resnet_layers, False).to(device)
    weights = torch.load(pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    test_files=glob.glob('./../frames/*/*.png')
    

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):
       
        dst='results'+file.replace('./../frames','').replace('.png','')
       
        if not os.path.exists(dst+'_disp.png'):
       
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            img = imread(file).astype(np.float32)

            h, w, _ = img.shape

            img = imresize(img, (img_height, img_width)).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))

            tensor_img = torch.from_numpy(img).unsqueeze(0)
            tensor_img = ((tensor_img/255 - 0.45)/0.225).to(device)

            output = disp_net(tensor_img)[0]


            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)


            depth = 1/output
            depth = (255*tensor2array(depth, max_value=1, colormap='magma')).astype(np.uint8)

            imsave(dst+'_disp.png', np.transpose(disp, (1, 2, 0)))
            imsave(dst+'_depth.png', np.transpose(depth, (1, 2, 0)))


