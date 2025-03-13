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
import random

# ======== PARAMETERS ========
IMG_SIZE = (150, 150)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'

###############################################################################
# 1. LOAD IMAGES AND EXTRACT HOG FEATURES
###############################################################################
def load_images_and_extract_features(folder):
    """
    Load all images from the given folder (for a single writer),
    convert to grayscale, resize, and extract HOG features.
    Returns:
      - features: list of HOG feature vectors
      - filenames: list of filenames for reference
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

###############################################################################
# 2. LOAD GENUINE & FORGED SIGNATURES (WRITER-DEPENDENT)
###############################################################################
genuine_folder = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_org"
forged_folder  = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\full_forg"

X_genuine, genuine_files = load_images_and_extract_features(genuine_folder)
X_forged, forged_files   = load_images_and_extract_features(forged_folder)

y_genuine = np.ones(len(X_genuine), dtype=int)
y_forged  = np.zeros(len(X_forged), dtype=int)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

X_all = np.concatenate([X_genuine, X_forged])
y_all = np.concatenate([y_genuine, y_forged])

print("Combined feature matrix shape:", X_all.shape)
print("Combined labels shape:", y_all.shape)

###############################################################################
# 3. RANDOMLY SAMPLE PAIRS
###############################################################################
def create_pairs_sampled(features, labels, max_pairs=5000):
    """
    Instead of creating all pairs (which can be huge), randomly sample 'max_pairs' pairs.
    For each randomly chosen pair (i, j), compute the absolute difference of feature vectors.
    Label is 1 if labels[i] == labels[j], else 0.
    """
    n = len(features)
    pairs = []
    pair_labels = []

    # If the number of possible unique pairs < max_pairs, just create them all
    total_possible_pairs = (n*(n-1)) // 2
    if total_possible_pairs <= max_pairs:
        print(f"Total possible pairs = {total_possible_pairs}; generating all pairs.")
        all_indices = []
        for i in range(n):
            for j in range(i+1, n):
                all_indices.append((i,j))
        random.shuffle(all_indices)
        for (i, j) in all_indices:
            diff = np.abs(features[i] - features[j])
            pairs.append(diff)
            pair_labels.append(1 if labels[i] == labels[j] else 0)
    else:
        print(f"Total possible pairs = {total_possible_pairs}; randomly sampling {max_pairs} pairs.")
        for _ in range(max_pairs):
            i, j = random.sample(range(n), 2)  # pick 2 distinct indices
            diff = np.abs(features[i] - features[j])
            pairs.append(diff)
            pair_labels.append(1 if labels[i] == labels[j] else 0)

    return np.array(pairs), np.array(pair_labels)

# You can adjust max_pairs as needed
pairs, pair_labels = create_pairs_sampled(X_all, y_all, max_pairs=5000)

print("Pairs shape:", pairs.shape, "Pair labels shape:", pair_labels.shape)

###############################################################################
# 4. SPLIT DATA (TRAIN/DEV/TEST)
###############################################################################
pairs_temp, pairs_test, labels_temp, labels_test = train_test_split(
    pairs, pair_labels, test_size=0.20, random_state=42)

pairs_train, pairs_dev, labels_train, labels_dev = train_test_split(
    pairs_temp, labels_temp, test_size=0.25, random_state=42)

print("Training set size:", pairs_train.shape[0])
print("Development set size:", pairs_dev.shape[0])
print("Test set size:", pairs_test.shape[0])

###############################################################################
# 5. TRAIN LOGISTIC REGRESSION
###############################################################################
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(pairs_train, labels_train)

# Evaluate on dev
y_dev_pred = model.predict(pairs_dev)
dev_accuracy = accuracy_score(labels_dev, y_dev_pred)
print("Development Set Accuracy (Logistic Regression): {:.2f}%".format(dev_accuracy * 100))
print("\nDevelopment Classification Report:")
print(classification_report(labels_dev, y_dev_pred))

# Evaluate on test
y_test_pred = model.predict(pairs_test)
test_accuracy = accuracy_score(labels_test, y_test_pred)
print("Test Set Accuracy (Logistic Regression): {:.2f}%".format(test_accuracy * 100))
print("\nTest Classification Report:")
print(classification_report(labels_test, y_test_pred))

###############################################################################
# 6. SAVE MODEL
###############################################################################
import joblib
joblib.dump(model, "writer_dependent_logreg_model.pkl")
print("Logistic Regression model saved as writer_dependent_logreg_model.pkl")

###############################################################################
# 7. CLASSIFY A SINGLE SIGNATURE PAIR (OPTIONAL)
###############################################################################
def classify_signature_pair(test_img_path, reference_img_path, model, threshold=0.5):
    """
    For a writer-dependent system, compare test signature vs. reference genuine signature.
    - Load & preprocess images
    - Extract HOG features
    - Compute abs diff
    - Model outputs similarity probability
    """
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    ref_img  = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None or ref_img is None:
        print("Error: Unable to load images.")
        return None

    test_img = cv2.resize(test_img, IMG_SIZE)
    ref_img  = cv2.resize(ref_img, IMG_SIZE)

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

    diff_features = np.abs(test_hog - ref_hog).reshape(1, -1)
    prob = model.predict_proba(diff_features)[0][1]
    classification = "Genuine" if prob >= threshold else "Forged"
    print(f"Test Signature: {os.path.basename(test_img_path)}")
    print(f"Reference Signature: {os.path.basename(reference_img_path)}")
    print(f"Predicted similarity probability: {prob:.2f}")
    print(f"Classification: {classification}")
    return classification, prob

# Example usage
test_signature_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_forg\forgeries_41_1.png"
reference_signature_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_org\original_41_1.png"

trained_model = joblib.load("writer_dependent_logreg_model.pkl")
classify_signature_pair(test_signature_path, reference_signature_path, trained_model, threshold=0.5)
