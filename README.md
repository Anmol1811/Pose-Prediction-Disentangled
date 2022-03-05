# Disentangled Pose Prediction

This repo contains the code to create and train models for pose prediction as described in the Project Report.

## Installation
------------
Start by cloning this repositiory:
```
git clone https://github.com/vita-epfl/pose-prediction-disentangled.git
cd pose-prediction-disentangled
```
Create a new conda environment (Python 3.7):
```
conda create -n pose-de python=3.7
conda activate pose-de
```
And install the dependencies:
```
pip install -r requirements.txt
```

- The following libraries are required to run this code:
Pytorch 1.4.0+, OpenCV, Numpy, Pandas, PIL, matplotlib, glob, json

- For running the SC-sfmlearner code, please refer to the original repo (https://github.com/JiawangBian/SC-SfMLearner-Release) to download models. 
To generate the depth and egomotion information, the code is provided in the SC-sfmlearner folder(preprocess_depth.py and preprocess_egomotion.py)

- For generating Openpose keypoint outputs, refer to original documentation (https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation)
- For generating OpenPifPaf keypoint outputs refer to documentation (https://openpifpaf.github.io/cli_help.html), specifically CLI using --glob option.

conda create -n pv-lstm python=3.7
conda activate pv-lstm

## Dataset
------------
  
  * Clone the dataset's [repository](https://github.com/ykotseruba/JAAD).
  ```
  git clone https://github.com/ykotseruba/JAAD
  ```
  * Run the `prepare_data.py` script, make sure you provide the path to the JAAD repository and the train/val/test ratios (ratios must be in [0,1] and their sum should equal 1.
  ```
  python3 prepare_data.py |path/to/JAAD/repo| |train_ratio| |val_ratio| |test_ratio|
  ```
  * Download the [JAAD clips](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) (UNRESIZED) and unzip them in the `videos` folder.
  * Run the script `split_clips_to_frames.sh` to convert the JAAD videos into frames. Each frame will be placed in a folder under the `scene` folder. Note that this takes 169G of space.
  * Run the openpifpaf/openpose keypoint detectors on the images to get the keypoint outputs 
  * The processed csv files for openpifpaf/openpose/posetrack/3dpw are in the processed_csvs folder.



## Usage
------------

1. The outputs of openpose and openpifpaf are json files, which can be parsed using create_data.py. This will create a pickle file containing all the keypoints with their video, frame and pedestrian numbers. There are then sorted accordingly and split into sequences which are saved as CSV files.

2. The posetrack and 3dpw files can be downloaded from the SoMoF benchmark site(https://somof.stanford.edu/dataset). Before running the model on it, it is required to convert it to the cvs format, which can be done by running the following command:
```
python3 pre_process_somof.py 
```

3. All the models are described in their individual files in the models folder. The dataloaders are defined according to the model type in utils.py - 

    - LSTM_vel, LSTM_posetrack and LSTM_3dpw use the standard data_loader class, which returns the observed/target poses and velocities

    - PV_LSTM_DE, LSTM_spat and LSTM_spat_rec use the data_loader_DE class, which splits the pose and returns the observed/target velocities and poses as local and global streams.

    - glob_LSTM_de uses another dataloader which is defined in the file itself.



Start training the network by running the command:
```
python3 train.py
```

Results can be visualized by running the command:
```
python3 test.py
```

## Tested Environments:
------------
  * Windows 10, CUDA 10.1