import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib  # <-- For model persistence

# HOG Parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
IMG_SIZE = (150, 150)

# Function to extract HOG features from an image
def load_images_and_extract_features(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            hog_features = hog(img,
                               orientations=orientations,
                               pixels_per_cell=pixels_per_cell,
                               cells_per_block=cells_per_block,
                               block_norm='L2-Hys')
            features.append(hog_features)
            labels.append(label)
    return np.array(features), np.array(labels)

genuine_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_org"
forged_path  = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_forg"

X_genuine, y_genuine = load_images_and_extract_features(genuine_path, label=1)
X_forged,  y_forged  = load_images_and_extract_features(forged_path, label=0)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

X = np.concatenate([X_genuine, X_forged])
y = np.concatenate([y_genuine, y_forged])

print("Combined feature matrix shape:", X.shape)
print("Combined labels shape:", y.shape)

# Dataset path for training the model
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print("Training set size:", X_train.shape[0])
print("Development set size:", X_dev.shape[0])
print("Test set size:", X_test.shape[0])

# Train the model
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train, y_train)

# Dev evaluation
y_dev_pred = svm.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print("Development Set Accuracy: {:.2f}%".format(dev_accuracy * 100))
print("\nDevelopment Classification Report:")
print(classification_report(y_dev, y_dev_pred))

# Test evaluation
y_test_pred = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Set Accuracy: {:.2f}%".format(test_accuracy * 100))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

joblib.dump(svm, "svm_model.pkl")
print("SVM model saved as svm_model.pkl")

# Function to compare HOG features of two images
def compare_hog(genuine_image_path, forged_image_path):
    genuine_img = cv2.imread(genuine_image_path, cv2.IMREAD_GRAYSCALE)
    forged_img  = cv2.imread(forged_image_path, cv2.IMREAD_GRAYSCALE)
    if genuine_img is None or forged_img is None:
        print("Error: One of the images couldn't be loaded.")
        return
    
    genuine_img = cv2.resize(genuine_img, IMG_SIZE)
    forged_img  = cv2.resize(forged_img, IMG_SIZE)
    
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

compare_hog(r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_org\original_41_1.png", 
            r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_forg\forgeries_41_1.png")
