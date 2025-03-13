import numpy as np
import os
import shutil

if os.path.exists('./ocean/data/dataset'):
    shutil.rmtree('./ocean/data/dataset')

# os.makedirs('./ocean/data/dataset')
# os.makedirs('./ocean/data/dataset/hr/psi1')
# os.makedirs('./ocean/data/dataset/lr/psi1')
# os.makedirs('./ocean/data/dataset/hr/psi2')
# os.makedirs('./ocean/data/dataset/lr/psi2')

folder_paths = [
    'ocean/data/dataset/hr/psi1/train_set/0',
    'ocean/data/dataset/lr/psi1/train_set/0',
    'ocean/data/dataset/hr/psi1/test_set/0',
    'ocean/data/dataset/lr/psi1/test_set/0',
    'ocean/data/dataset/hr/psi2/train_set/0',
    'ocean/data/dataset/lr/psi2/train_set/0',
    'ocean/data/dataset/hr/psi2/test_set/0',
    'ocean/data/dataset/lr/psi2/test_set/0'
]

for folder_path in folder_paths:
    # if dataset folder exists, delete it
    if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    # create train and test set folders for each psi and resolution
    os.makedirs(folder_path)

hr_folder_path = './ocean/data/sr_dataset/hr_320_544'
paths = os.listdir(hr_folder_path)
n = np.load(hr_folder_path + '/' + paths[0]).shape[0]/2
ns = [i for i in range(int(n))]
test_set_idx = np.random.choice(ns, int(150*0.2), replace=False)
for i, path in enumerate(paths):
    arr = np.load(hr_folder_path + '/' + path)
    for j, sample in enumerate(arr):
        sample = sample.reshape(544, 320)
        if j < 150:
            psi = 'psi1'
        else:
            psi = 'psi2'

        if j % 150 in(test_set_idx):
            set = 'test_set'
        else:
            set = 'train_set' 

        np.save(f'./ocean/data/dataset/hr/{psi}/{set}/0/{i}_{j}.npy', sample)

lr_folder_path = './ocean/data/sr_dataset/lr_40_68'
paths = os.listdir(lr_folder_path)
for i, path in enumerate(paths):
    arr = np.load(lr_folder_path + '/' + path)
    for j, sample in enumerate(arr):
        sample = sample.reshape(68, 40)
        if j < 150:
            psi = 'psi1'
        else:
            psi = 'psi2'

        if j % 150 in(test_set_idx):
            set = 'test_set'
        else:
            set = 'train_set' 

        np.save(f'./ocean/data/dataset/lr/{psi}/{set}/0/{i}_{j}.npy', sample)