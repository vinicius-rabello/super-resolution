import os
import numpy as np
import cv2
import shutil

# if dataset folder exists, delete it
if os.path.exists('ocean/data/dataset'):
        shutil.rmtree('ocean/data/dataset')
# create train and test set folders for each psi
os.makedirs('ocean/data/dataset/psi1/train_set/0')
os.makedirs('ocean/data/dataset/psi1/test_set/0')
os.makedirs('ocean/data/dataset/psi2/train_set/0')
os.makedirs('ocean/data/dataset/psi2/test_set/0')

# iterate over each psi
for psi in ['psi1', 'psi2']:
    folder_path = f"ocean/data/processed/{psi}"
    paths = os.listdir(folder_path) # get every array path in psi folder
    np.random.shuffle(paths) # randomly shuffle the paths
    train_split = int(len(paths)*0.8) # divide into train and test sets
    train_set_paths = paths[:train_split]
    test_set_paths = paths[train_split:]

    # saving each image to the corresponding folder
    print('creating training set...')    
    for i, array in enumerate(train_set_paths):
        array = np.load(folder_path + '/' + array)
        print(f'saving {i}.png to data/dataset/{psi}/train_set/0...')
        cv2.imwrite(f'ocean/data/dataset/{psi}/train_set/0/{i}.png', 255*array)
        
    print('creating test set...')
    for i, array in enumerate(test_set_paths):
        array = np.load(folder_path + '/' + array)
        print(f'saving {i + train_split}.png to data/dataset/{psi}/test_set/0...')
        cv2.imwrite(f'ocean/data/dataset/{psi}/test_set/0/{i + train_split}.png', 255*array)
        
    print('dataset is ready!')