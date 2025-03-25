import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

# Function to load and preprocess images
def load_images(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    images = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path)  # Load color image
        img = cv2.resize(img, (128, 128))  # Resize for consistency
        img = img.flatten()  # Convert to a 1D feature vector
        images.append(img)

    return np.array(images), image_paths

# Path to cropped images folder
image_folder = "/home5/ayesha.cse/mycode/data/benigncrops"  # Change this to your actual folder path
image_features, image_paths = load_images(image_folder)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=8000, min_samples=1, metric='euclidean')  # Adjust eps based on your dataset
labels = dbscan.fit_predict(image_features)

# Print clustering results
for img_path, label in zip(image_paths, labels):
    print(f"Image: {img_path} -> Cluster: {label}")

# (Optional) Save clustered images into separate folders
output_folder = "clustered_images"
os.makedirs(output_folder, exist_ok=True)

for img_path, label in zip(image_paths, labels):
    cluster_folder = os.path.join(output_folder, f"cluster_{label}")
    os.makedirs(cluster_folder, exist_ok=True)
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(cluster_folder, img_name), cv2.imread(img_path))

print("Clustered images saved in separate folders.")
