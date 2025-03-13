# writer_dependent_signature_verification.py

import os
import numpy as np
import cv2
import re
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

# ======== FUNCTION: Load images and extract HOG features ========
def load_images_and_extract_features(folder):
    """
    Load all images from the given folder,
    convert to grayscale, resize, and extract HOG features.
    Returns a list of HOG feature vectors and a list of filenames.
    """
    features = []
    filenames = []
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
            filenames.append(filename)
    return np.array(features), filenames

# ======== Paths to your datasets for one writer ========
# (Assume both folders contain signatures for the same writer)
genuine_folder = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_org"
forged_folder  = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_forg"

# Load features from genuine and forged folders
X_genuine, genuine_files = load_images_and_extract_features(genuine_folder)
X_forged, forged_files = load_images_and_extract_features(forged_folder)

# Create labels: genuine = 1, forged = 0
y_genuine = np.ones(len(X_genuine), dtype=int)
y_forged  = np.zeros(len(X_forged), dtype=int)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

# ======== Combine Data ============
# In writer-dependent verification, we assume that all images belong to the same writer.
# We will later create pairs of images.
X_all = np.concatenate([X_genuine, X_forged])
y_all = np.concatenate([y_genuine, y_forged])

print("Combined feature matrix shape:", X_all.shape)
print("Combined labels shape:", y_all.shape)

# ======== Create Pairs and Difference Features ============
def create_pairs(features, labels):
    """
    Creates all unique pairs from a set of feature vectors.
    For each pair, compute the absolute difference (element-wise).
    Label the pair as 1 if both images have the same label, else 0.
    """
    pairs = []
    pair_labels = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.abs(features[i] - features[j])
            pairs.append(diff)
            pair_labels.append(1 if labels[i] == labels[j] else 0)
    return np.array(pairs), np.array(pair_labels)

pairs, pair_labels = create_pairs(X_all, y_all)
print("Pairs shape:", pairs.shape, "Pair labels shape:", pair_labels.shape)

# ======== Split Data ============
# First, split off 20% for testing
pairs_temp, pairs_test, labels_temp, labels_test = train_test_split(pairs, pair_labels, test_size=0.20, random_state=42)
# Then split the remaining 80% into training (≈75% of 80% ≈ 60% overall) and development (≈25% of 80% ≈ 20% overall)
pairs_train, pairs_dev, labels_train, labels_dev = train_test_split(pairs_temp, labels_temp, test_size=0.25, random_state=42)

print("Training set size:", pairs_train.shape[0])
print("Development set size:", pairs_dev.shape[0])
print("Test set size:", pairs_test.shape[0])

# ======== Train Logistic Regression Classifier ============
# Train the model on the difference features between pairs.
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(pairs_train, labels_train)

# Evaluate on development set
y_dev_pred = model.predict(pairs_dev)
dev_accuracy = accuracy_score(labels_dev, y_dev_pred)
print("Development Set Accuracy (Logistic Regression): {:.2f}%".format(dev_accuracy * 100))
print("\nDevelopment Classification Report (Logistic Regression):")
print(classification_report(labels_dev, y_dev_pred))

# Evaluate on test set
y_test_pred = model.predict(pairs_test)
test_accuracy = accuracy_score(labels_test, y_test_pred)
print("Test Set Accuracy (Logistic Regression): {:.2f}%".format(test_accuracy * 100))
print("\nTest Classification Report (Logistic Regression):")
print(classification_report(labels_test, y_test_pred))

# Save the trained logistic regression model
joblib.dump(model, "writer_dependent_logreg_model.pkl")
print("Logistic Regression model saved as writer_dependent_logreg_model.pkl")

# ======== Function: Classify a Single Signature Pair ============
def classify_signature_pair(test_img_path, reference_img_path, model, threshold=0.5):
    """
    For a writer-dependent system, compare a test signature with a reference genuine signature.
    - Load and preprocess both images.
    - Extract HOG features.
    - Compute the absolute difference between their feature vectors.
    - Use the trained logistic regression model to predict the probability.
    If the predicted probability is ≥ threshold, the signatures are considered similar (genuine);
    otherwise, the test signature is flagged as a forgery.
    """
    # Load test image and reference image in grayscale
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None or ref_img is None:
        print("Error: Unable to load one or both images.")
        return None

    # Resize images
    test_img = cv2.resize(test_img, IMG_SIZE)
    ref_img = cv2.resize(ref_img, IMG_SIZE)
    
    # Extract HOG features
    test_hog = hog(test_img,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm=block_norm)
    ref_hog = hog(ref_img,
                  orientations=orientations,
                  pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block,
                  block_norm=block_norm)
    # Compute absolute difference
    diff_features = np.abs(test_hog - ref_hog).reshape(1, -1)
    # Predict using logistic regression
    prob = model.predict_proba(diff_features)[0][1]  # probability that the pair is labeled as "same"
    classification = "Genuine" if prob >= threshold else "Forged"
    print(f"Test Signature: {os.path.basename(test_img_path)}")
    print(f"Reference Signature: {os.path.basename(reference_img_path)}")
    print(f"Predicted similarity probability: {prob:.2f}")
    print(f"Classification: {classification}")
    return classification, prob

# ======== Example Inference ============
# Replace these with your actual image file paths for a particular writer.
test_signature_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_forg\forgeries_41_1.png"
reference_signature_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_org\original_41_1.png"

# Load the trained model
trained_model = joblib.load("writer_dependent_logreg_model.pkl")
classify_signature_pair(test_signature_path, reference_signature_path, trained_model, threshold=0.5)
