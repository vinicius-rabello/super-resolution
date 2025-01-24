import cv2
import os

folder_path = '../data/synthetic/raw'
img_paths = os.listdir(folder_path)

for img_path in img_paths:
    print(img_path)