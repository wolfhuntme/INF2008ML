# ensemble.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import joblib

# ======== PARAMETERS ========
IMG_SIZE = (150, 150)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'
THRESHOLD = 0.5  # if ensemble genuine probability is >= 0.5, classify as "Not Fraud" (genuine)

# ======== FUNCTION: Extract HOG Features ============
def extract_hog_features(image_path):
    """
    Loads an image, converts it to grayscale, resizes to IMG_SIZE, and extracts HOG features.
    Returns the feature vector reshaped to (1, -1).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Unable to load image at {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    features = hog(img,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm=block_norm)
    return features.reshape(1, -1)

# ======== FUNCTION: Ensemble Classification ============
def ensemble_classify_signature(image_path, models, threshold=THRESHOLD):
    """
    Classify a single signature image using an ensemble of models.
    Each model returns the probability for the genuine class.
    The ensemble probability is the average of these probabilities.
    If the ensemble probability is >= threshold, classify as "Not Fraud (Genuine)",
    otherwise as "Fraud (Forged)".
    """
    features = extract_hog_features(image_path)
    probs = []
    for model in models:
        prob = model.predict_proba(features)[0][1]  # probability for genuine class
        probs.append(prob)
    ensemble_prob = np.mean(probs)
    classification = "Not Fraud (Genuine)" if ensemble_prob >= threshold else "Fraud (Forged)"
    return classification, ensemble_prob

# ======== MAIN SCRIPT ============
def main():
    # Load the pre-trained models (update the paths as needed)
    knn_model_path = r"C:\Users\xavie\Desktop\School\INF2008ML\knn_model.pkl"
    rf_model_path = r"C:\Users\xavie\Desktop\School\INF2008ML\random_forest_model.pkl"
    svm_model_path = r"C:\Users\xavie\Desktop\School\INF2008ML\svm_model.pkl"
    logreg_model_path = r"C:\Users\xavie\Desktop\School\INF2008ML\logreg_model.pkl"
    
    knn = joblib.load(knn_model_path)
    rf = joblib.load(rf_model_path)
    svm = joblib.load(svm_model_path)
    logreg = joblib.load(logreg_model_path)
    
    models = [knn, rf, svm, logreg]
    
    # Update the test image path to the signature you wish to classify
    test_image_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_forg\forgeries_43_1.png"
    
    classification, ensemble_prob = ensemble_classify_signature(test_image_path, models, threshold=THRESHOLD)
    print("Ensemble Classification:", classification)
    print("Ensemble Genuine Probability: {:.2f}".format(ensemble_prob))
    
    # Optional: display the test image with its classification result
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        plt.imshow(img, cmap="gray")
        plt.title(f"Test Image - Classified as: {classification}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
