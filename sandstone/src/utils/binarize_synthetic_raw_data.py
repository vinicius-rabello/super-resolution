import cv2
import os

# Define the folder paths
folder_path = 'sandstone/data/synthetic/raw'
processed_folder_path = 'sandstone/data/synthetic/processed'

# Create the processed folder if it doesn't exist
if not os.path.exists(processed_folder_path):
    os.makedirs(processed_folder_path)

# List all image paths in the raw folder
img_paths = os.listdir(folder_path)

# Process each image
for img_name in img_paths:
    # Construct the full path to the raw image
    raw_img_path = os.path.join(folder_path, img_name)
    
    # Read the image in grayscale mode
    img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarize the image using a threshold of 255/2
    _, binary_img = cv2.threshold(img, 255/2, 255, cv2.THRESH_BINARY)
    
    # Construct the full path to save the processed image
    processed_img_path = os.path.join(processed_folder_path, img_name)
    
    # Save the binarized image
    cv2.imwrite(processed_img_path, binary_img)
    
    print(f"Processed and saved: {processed_img_path}")