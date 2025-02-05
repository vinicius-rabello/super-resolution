import cv2
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define the folder paths
folder_path = 'sandstone/data/synthetic/raw'
processed_folder_path = 'sandstone/data/synthetic/processed'

# Resets the processed folder if it already exists
if os.path.exists(processed_folder_path):
    shutil.rmtree(processed_folder_path)
os.makedirs(processed_folder_path)

# Create the train and test folders if they don't exist
train_folder = os.path.join(processed_folder_path, 'train_set', '0')
test_folder = os.path.join(processed_folder_path, 'test_set', '0')

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# List all image paths in the raw folder
img_paths = os.listdir(folder_path)

# Split the images into train and test sets (80% train, 20% test)
train_img_paths, test_img_paths = train_test_split(img_paths, train_size=0.8, random_state=42)

# Function to process and save images
def process_and_save_images(img_names, output_folder):
    for img_name in img_names:
        # Construct the full path to the raw image
        raw_img_path = os.path.join(folder_path, img_name)
        
        # Read the image in grayscale mode
        img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Binarize the image using a threshold of 255/2
        _, binary_img = cv2.threshold(img, 255/2, 255, cv2.THRESH_BINARY)
        
        # Construct the full path to save the processed image
        processed_img_path = os.path.join(output_folder, img_name)
        
        # Save the binarized image
        cv2.imwrite(processed_img_path, binary_img)
        
        print(f"Processed and saved: {processed_img_path}")

# Process and save train images
process_and_save_images(train_img_paths, train_folder)

# Process and save test images
process_and_save_images(test_img_paths, test_folder)