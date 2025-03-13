import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ======== PARAMETERS ========
# HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# Image size to which all signatures are resized
IMG_SIZE = (150, 150)

# ======== FUNCTION: Load images and extract HOG features ========
def load_images_and_extract_features(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            # Read the image in grayscale
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip files that cannot be read as images
            # Resize the image to a uniform size
            img = cv2.resize(img, IMG_SIZE)
            # Extract HOG features
            hog_features = hog(img,
                               orientations=orientations,
                               pixels_per_cell=pixels_per_cell,
                               cells_per_block=cells_per_block,
                               block_norm='L2-Hys')
            features.append(hog_features)
            labels.append(label)
    return np.array(features), np.array(labels)

# ======== Paths to your datasets ========
genuine_path = r"signatures_cedar/full_org"  # Update this path
forged_path  = r"signatures_cedar/full_forg"  # Update this path

# ======== Load and process data ========
X_genuine, y_genuine = load_images_and_extract_features(genuine_path, label=1)
X_forged,  y_forged  = load_images_and_extract_features(forged_path, label=0)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

# Combine datasets
X = np.concatenate([X_genuine, X_forged])
y = np.concatenate([y_genuine, y_forged])

print("Combined feature matrix shape:", X.shape)
print("Combined labels shape:", y.shape)

# ======== Three-Way Split ========
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print("Training set size:", X_train.shape[0])
print("Development set size:", X_dev.shape[0])
print("Test set size:", X_test.shape[0])

# ======== Train the Random Forest classifier ========
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators
rf.fit(X_train, y_train)

# ======== Evaluate on Development Set ========
y_dev_pred_rf = rf.predict(X_dev)
dev_accuracy_rf = accuracy_score(y_dev, y_dev_pred_rf)
print("Development Set Accuracy (Random Forest): {:.2f}%".format(dev_accuracy_rf * 100))
print("\nDevelopment Classification Report (Random Forest):")
print(classification_report(y_dev, y_dev_pred_rf))

# ======== Evaluate on Test Set ========
y_test_pred_rf = rf.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
print("Test Set Accuracy (Random Forest): {:.2f}%".format(test_accuracy_rf * 100))
print("\nTest Classification Report (Random Forest):")
print(classification_report(y_test, y_test_pred_rf))

# ======== Optional: Visualize HOG features for a sample image ========
def compare_hog(genuine_image_path, forged_image_path):
    # Load images in grayscale
    genuine_img = cv2.imread(genuine_image_path, cv2.IMREAD_GRAYSCALE)
    forged_img  = cv2.imread(forged_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if images are loaded
    if genuine_img is None or forged_img is None:
        print("Error: One of the images couldn't be loaded. Check the file paths.")
        return
    
    # Resize images to IMG_SIZE
    genuine_img = cv2.resize(genuine_img, IMG_SIZE)
    forged_img  = cv2.resize(forged_img, IMG_SIZE)
    
    # Extract HOG features with visualization enabled
    genuine_features, genuine_hog_image = hog(genuine_img,
                                              orientations=9,
                                              pixels_per_cell=(8,8),
                                              cells_per_block=(2,2),
                                              block_norm='L2-Hys',
                                              visualize=True)
    
    forged_features, forged_hog_image = hog(forged_img,
                                            orientations=9,
                                            pixels_per_cell=(8,8),
                                            cells_per_block=(2,2),
                                            block_norm='L2-Hys',
                                            visualize=True)
    
    # Plot the results side by side
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(genuine_img, cmap='gray')
    axes[0, 0].set_title("Genuine Original")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(genuine_hog_image, cmap='gray')
    axes[0, 1].set_title("Genuine HOG")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(forged_img, cmap='gray')
    axes[1, 0].set_title("Forged Original")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(forged_hog_image, cmap='gray')
    axes[1, 1].set_title("Forged HOG")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage (update the paths accordingly)
compare_hog(r"signatures_cedar/unseen_data_for_testing/unseen_org/original_43_7.png", 
            r"signatures_cedar/unseen_data_for_testing/unseen_forg/forgeries_41_2.png")
