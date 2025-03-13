# writer_independent_logistic_regression.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======== PARAMETERS ========
IMG_SIZE = (150, 150)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'

###############################################################################
# 1. LOAD IMAGES AND EXTRACT HOG FEATURES
###############################################################################
def load_images_and_extract_features(folder, label):
    """
    Load all images from the given folder,
    convert to grayscale, resize, and extract HOG features.
    Returns:
      - features: list of HOG feature vectors
      - labels: list of labels (1 for genuine, 0 for forged)
    """
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
                               block_norm=block_norm)
            features.append(hog_features)
            labels.append(label)
    return np.array(features), np.array(labels)

###############################################################################
# 2. LOAD GENUINE & FORGED SIGNATURES FROM MULTIPLE WRITERS
###############################################################################
# Update these paths to point to your datasets
genuine_folder = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_org"
forged_folder  = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_forg"

X_genuine, _ = load_images_and_extract_features(genuine_folder, label=1)
X_forged,  _ = load_images_and_extract_features(forged_folder, label=0)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

# Combine data from multiple writers
X_all = np.concatenate([X_genuine, X_forged])
y_all = np.concatenate([np.ones(len(X_genuine), dtype=int), np.zeros(len(X_forged), dtype=int)])

print("Combined feature matrix shape:", X_all.shape)
print("Combined labels shape:", y_all.shape)

###############################################################################
# 3. SPLIT DATA INTO TRAIN, DEV, TEST (Writer-Independent)
###############################################################################
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print("Training set size:", X_train.shape[0])
print("Development set size:", X_dev.shape[0])
print("Test set size:", X_test.shape[0])

###############################################################################
# 4. TRAIN LOGISTIC REGRESSION CLASSIFIER
###############################################################################
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate on development set
y_dev_pred = model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print("Development Set Accuracy (Logistic Regression): {:.2f}%".format(dev_accuracy * 100))
print("\nDevelopment Classification Report:")
print(classification_report(y_dev, y_dev_pred))

# Evaluate on test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Set Accuracy (Logistic Regression): {:.2f}%".format(test_accuracy * 100))
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

###############################################################################
# 5. SAVE THE TRAINED MODEL FOR PERSISTENCE
###############################################################################
joblib.dump(model, "writer_independent_logreg_model.pkl")
print("Logistic Regression model saved as writer_independent_logreg_model.pkl")

###############################################################################
# 6. CLASSIFY A SINGLE SIGNATURE IMAGE (OPTIONAL)
###############################################################################
def classify_single_signature(image_path, model, threshold=0.5):
    """
    Classify a single signature image as Genuine or Forged.
    This is writer-independent so the model directly predicts based on its HOG features.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    img = cv2.resize(img, IMG_SIZE)
    hog_features = hog(img,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm=block_norm)
    hog_features = hog_features.reshape(1, -1)
    pred_prob = model.predict_proba(hog_features)[0][1]
    classification = "Genuine" if pred_prob >= threshold else "Forged"
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted genuine probability: {pred_prob:.2f}")
    print(f"Classification: {classification}")
    return classification, pred_prob

# Example usage (update the path accordingly)
test_image_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_org\original_41_1.png"
classify_single_signature(test_image_path, model, threshold=0.5)
