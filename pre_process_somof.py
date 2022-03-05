import json

with open("./Posetrack/posetrack_train_in.json", "r") as read_file:
    data = json.load(read_file)
posesin=[]
for i in data:
    for j in i:
        posesin.append(j)


with open("./Posetrack/posetrack_train_masks_in.json", "r") as read_file:
    data = json.load(read_file)
masksin=[]
for i in data:
    for j in i:
        masksin.append(j)

with open("./Posetrack/posetrack_train_out.json", "r") as read_file:
    data = json.load(read_file)
      
posesout=[]
for i in data:
    for j in i:
        posesout.append(j)


with open("./Posetrack/posetrack_train_masks_out.json", "r") as read_file:
    data = json.load(read_file)
      
maskout=[]
for i in data:
    for j in i:
        maskout.append(j)



data_train = {'Pose': posesin,
            'Future_Pose': posesout,
            'Mask': masksin,
            'Future_Mask': masksout
        }
df = pd.DataFrame (data_train, columns = ['Pose','Future_Pose','Mask','Future_Mask'])
df.to_csv("./posetrack_train.csv",index=False)


