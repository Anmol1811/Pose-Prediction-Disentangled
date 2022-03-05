import json
import glob 
import pickle


keypoints=[]
#rename folder accordingly
folder='openpose/openpifpaf_jsons/*'

keypoints={}
cm=0
for c,i in enumerate(glob.glob(folder)):
  if(c%10000)==0:
        print(c,cm)
  with open(i) as json_file:
    data = json.load(json_file)

    if(len(data)>1):
        cm+=1
    if (data and data[0]['keypoints']):
      keypoints[i]=data[0]['keypoints']
print(cm)


pickle.dump(keypoints,open('./keypoints_openpifpaf.pickle','wb'))

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
    
#   if np.count_nonzero(kpd[key]<=0)<=9:
#       kpa.append([vid_num,frame_num,ped_num,kpd[key]])
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
print(kp_train.shape)

import itertools

key_f = lambda e: (e[0],e[2])
kp_seq=[]
for key, group in itertools.groupby(keypoints_array_sorted, key_f):
    lg=list(group)
    # print(len(lg))
    kp_seq.append(lg)

print(len(keypoints_array_sorted))
gps=[]
for seq in kp_seq:
  l=[]
  for i in range(len(seq)):
#     print(seq[i][3].shape)
    vec=seq[i][3][2::3]
    if (vec > 0.25).all():
      l.append(keypoints_array_sorted[i][3])
  gps.append(l)

gps = list(filter(None, gps))
print(len(gps))

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data[:,1]) != stepsize)[0]+1)

print(len(kp_seq))
sequences=[]
seq_len=16
slide=4
c=0
for lst in kp_seq:
  c+=1
  ll=np.array(lst)
  x=[np.array(i) for i in consecutive(ll)]
  
  for tt in x:
    if(len(tt)>=seq_len):

      pp=[[tt[i:i + seq_len],tt[i+seq_len:i + 2*seq_len]] for i in range(0, len(tt)-(2*seq_len), slide)]
      sequences.extend(pp)

print(len(sequences))

array_of_seq=[]
tuples_to_save=[]
tups_x=[]
tups_y=[]
array_of_data=[]
for i in sequences:
  # print(len(i[0]),len(i[1]))
  x=np.array([p[3] for p in i[0]])
  y=np.array([p[3] for p in i[1]])
  array_of_seq.append([x,y])
  tuples_to_save.extend([(p[0],p[1],p[2]) for p in i[0]])
  tuples_to_save.extend([(p[0],p[1],p[2]) for p in i[1]])
#     tups_y.append()
  array_of_data.append([x,y,np.array([(p[0],p[1],p[2]) for p in i[0]]),np.array([(p[0],p[1],p[2]) for p in i[1]])])
  
array_of_seq=np.array(array_of_seq)
print(array_of_seq.shape)
# print(len(tuples_to_save),tuples_to_save[0])
# array_of_data[0]


import random 
seq_out=array_of_seq
print(seq_out.shape)
sequences_all=[]
sequences_obs_speed=[]
sequences_true_speed=[]
sequences_obs_pose=[]
sequences_true_pose=[]
sequences_true_imgs=[]
sequences_obs_imgs=[]
seq_len=16
split=int(0.75*len(seq_out))
for p in range(len(seq_out)):
  outputs = []

  observed = array_of_data[p][0]
  future = array_of_data[p][1]
  obs = torch.tensor([observed[i] for i in range(0,seq_len,1)])
  true = torch.tensor([future[i] for i in range(0,seq_len,1)])
  sequences_obs_pose.append(np.round(obs.numpy(),2).tolist())
  sequences_true_pose.append(np.round(true.numpy(),2).tolist())
  
c = list(zip(sequences_obs_pose, sequences_true_pose))#,sequences_true_imgs,sequences_obs_imgs))

random.shuffle(c)

sequences_obs_pose, sequences_true_pose = zip(*c) 

sequences_all_train=sequences_all[:split]
sequences_all_val=sequences_all[split:]

sequences_all=np.array(sequences_all)

data = {'Pose': sequences_obs_pose,[:split]
        'Future_Pose': sequences_true_pose,[:split]
       }

data_val = {'Pose': sequences_obs_pose[split:],
        'Future_Pose': sequences_true_pose[split:],
           }

df_train = pd.DataFrame (data, columns = ['Pose','Future_Pose'])
df_val = pd.DataFrame (data_val, columns = ['Pose','Future_Pose'])

df_train.to_csv("./sequences_openpifpaf_train.csv",index=False)
df_val.to_csv("./sequences_openpifpaf_val.csv", index=False)

print(df_train.head())