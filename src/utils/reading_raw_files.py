# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import shutil

# creating a list containg all paths to raw data
paths = os.listdir('data/raw')
paths = ['data/raw/' + path for path in paths]

# from each raw file, extract 1000 images and store them in a list
dataset = []
for path in paths:
    # opens the .raw file
    with open(path, 'rb') as f:
        unshaped_voxel = np.fromfile(f, dtype=np.uint8)
    # reshape it to 1000, 1000x1000 images
    images = unshaped_voxel.reshape(1000, 1000, 1000)
    dataset.extend(images)

# if dataset folder exists, delete it
if os.path.exists('data/dataset'):
        shutil.rmtree('data/dataset')
# create train and test set folders
os.makedirs('data/dataset/train_set/0')
os.makedirs('data/dataset/test_set/0')

# randomly shuffle the order of images, then do a 80/20 split of training and testing data
np.random.shuffle(dataset)
train_split = int(len(dataset)*0.8) # this number is 80% of the dataset length
train_set = dataset[:train_split]
test_set = dataset[train_split:]

# saving each image to the corresponding folder
print('creating training set...')    
for i, image in enumerate(train_set):
    print(f'saving {i}.png to data/dataset/train_set/0...')
    cv2.imwrite(f'data/dataset/train_set/0/{i}.png', 255*image)
    
print('creating test set...')
for i, image in enumerate(test_set):
    print(f'saving {i + train_split}.png to data/dataset/test_set/0...')
    cv2.imwrite(f'data/dataset/test_set/0/{i + train_split}.png', 255*image)
    
print('dataset is ready!')