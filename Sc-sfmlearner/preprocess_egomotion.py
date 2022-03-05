import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import numpy as np
from inverse_warp import pose_vec2mat
from scipy.ndimage.interpolation import zoom
from inverse_warp import *
import models
from imageio import imread, imsave
from skimage.transform import resize as imresize
from tqdm import tqdm

img_height=256
img_width=832
img_exts=['png', 'jpg', 'bmp']

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pretrained_posenet='./checkpoints/resnet50_pose_256/exp_pose_model_best.pth.tar'
def load_tensor_image(filename):
    img = imread('./../'+filename).astype(np.float32)
    h, w, _ = img.shape
    img = imresize(img, (img_height, img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img


with torch.no_grad():

    weights_pose = torch.load(pretrained_posenet)
    pose_net = models.PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    for i in tqdm(range(len(outputs))):

        test_files=outputs[i]    
        test_files.sort()
        dst='./egomotion/'+test_files[0].replace('/frames/','').replace('/','-').replace('.png','_to_')+test_files[-1].replace('/frames/','').replace('/','-').replace('.png','.txt')

        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            global_pose = np.eye(4)
            poses = [global_pose[0:3, :].reshape(1, 12)]

            n = len(test_files)
            tensor_img1 = load_tensor_image(test_files[0])
    
            for iter in range(n - 1):

                tensor_img2 = load_tensor_image(test_files[iter+1])

                pose = pose_net(tensor_img1, tensor_img2)

                pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
                pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
                global_pose = global_pose @  np.linalg.inv(pose_mat)

                poses.append(global_pose[0:3, :].reshape(1, 12))

                # update
                tensor_img1 = tensor_img2

            poses = np.concatenate(poses, axis=0)
            np.savetxt(dst, poses, delimiter=' ', fmt='%1.8e')
        
