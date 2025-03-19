import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tabulate import tabulate

# ========== PARAMETERS ==========
IMG_SIZE = (150, 150)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'

def preprocess_image(image_path):
    """Load an image, convert to grayscale, resize, and extract HOG features."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    features = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm
    )
    return features

def load_unseen_data(genuine_folder, forged_folder):
    """Load unseen images from genuine and forged folders, extract features, and assign labels."""
    X_unseen = []
    y_unseen = []
    
    # Genuine images (label = 1)
    for filename in os.listdir(genuine_folder):
        file_path = os.path.join(genuine_folder, filename)
        if os.path.isfile(file_path):
            try:
                features = preprocess_image(file_path)
                X_unseen.append(features)
                y_unseen.append(1)
            except Exception as e:
                print(e)
    
    # Forged images (label = 0)
    for filename in os.listdir(forged_folder):
        file_path = os.path.join(forged_folder, filename)
        if os.path.isfile(file_path):
            try:
                features = preprocess_image(file_path)
                X_unseen.append(features)
                y_unseen.append(0)
            except Exception as e:
                print(e)
    
    return np.array(X_unseen), np.array(y_unseen)

# ========== PATHS FOR UNSEEN DATA ==========
genuine_unseen_folder = r"C:\Users\Vyse\Documents\GitHub\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_org"   # Update this path
forged_unseen_folder  = r"C:\Users\Vyse\Documents\GitHub\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_forg"    # Update this path

# Load unseen data and extract features
X_unseen, y_unseen = load_unseen_data(genuine_unseen_folder, forged_unseen_folder)
print("Unseen data samples:", X_unseen.shape)

# ========== LOAD THE TRAINED AdaBoost MODEL ==========
adaboost_model = joblib.load("adaboost_model.pkl")
print("AdaBoost model loaded.")

# ========== EVALUATE THE AdaBoost MODEL ON UNSEEN DATA ==========
adaboost_predictions = adaboost_model.predict(X_unseen)
adaboost_accuracy = accuracy_score(y_unseen, adaboost_predictions)
adaboost_report = classification_report(y_unseen, adaboost_predictions, output_dict=True)
adaboost_precision = adaboost_report['1']['precision']
adaboost_recall = adaboost_report['1']['recall']
adaboost_f1 = adaboost_report['1']['f1-score']

# ========== LOAD THE TRAINED RandomForest MODEL ==========
randomForest_model = joblib.load("random_forest_model.pkl")
print("Random Forest model loaded.")

# ========== EVALUATE THE RandomForest MODEL ON UNSEEN DATA ==========
randomForest_predictions = randomForest_model.predict(X_unseen)
randomForest_accuracy = accuracy_score(y_unseen, randomForest_predictions)
randomForest_report = classification_report(y_unseen, randomForest_predictions, output_dict=True)
randomForest_precision = randomForest_report['1']['precision']
randomForest_recall = randomForest_report['1']['recall']
randomForest_f1 = randomForest_report['1']['f1-score']

# ========== LOAD THE TRAINED KNN MODEL ==========
KNN_model = joblib.load("knn_model.pkl")
print("KNN model loaded.")

# ========== EVALUATE THE KNN MODEL ON UNSEEN DATA ==========
KNN_predictions = KNN_model.predict(X_unseen)
KNN_accuracy = accuracy_score(y_unseen, KNN_predictions)
KNN_report = classification_report(y_unseen, KNN_predictions, output_dict=True)
KNN_precision = KNN_report['1']['precision']
KNN_recall = KNN_report['1']['recall']
KNN_f1 = KNN_report['1']['f1-score']

# ========== LOAD THE TRAINED SVM MODEL ==========
svm_model = joblib.load("svm_model.pkl")
print("SVM model loaded.")

# ========== EVALUATE THE SVM MODEL ON UNSEEN DATA ==========
svm_predictions = svm_model.predict(X_unseen)
svm_accuracy = accuracy_score(y_unseen, svm_predictions)
svm_report = classification_report(y_unseen, svm_predictions, output_dict=True)
svm_precision = svm_report['1']['precision']
svm_recall = svm_report['1']['recall']
svm_f1 = svm_report['1']['f1-score']

# ========== LOAD THE TRAINED LogReg MODEL ==========
LogReg_model = joblib.load("writer_independent_logreg_model.pkl")
print("LogReg model loaded.")

# ========== EVALUATE THE LogReg MODEL ON UNSEEN DATA ==========
LogReg_predictions = LogReg_model.predict(X_unseen)
LogReg_accuracy = accuracy_score(y_unseen, LogReg_predictions)
LogReg_report = classification_report(y_unseen, LogReg_predictions, output_dict=True)
LogReg_precision = LogReg_report['1']['precision']
LogReg_recall = LogReg_report['1']['recall']
LogReg_f1 = LogReg_report['1']['f1-score']

# Now store the results in a list for tabulation
model_comparison = [
    ["AdaBoost", round(adaboost_accuracy, 6), round(adaboost_precision, 6), round(adaboost_recall, 6), round(adaboost_f1, 6)],
    ["Random Forest", round(randomForest_accuracy, 6), round(randomForest_precision, 6), round(randomForest_recall, 6), round(randomForest_f1, 6)],
    ["KNN", round(KNN_accuracy, 6), round(KNN_precision, 6), round(KNN_recall, 6), round(KNN_f1, 6)],
    ["SVM", round(svm_accuracy, 6), round(svm_precision, 6), round(svm_recall, 6), round(svm_f1, 6)],
    ["LogReg", round(LogReg_accuracy, 6), round(LogReg_precision, 6), round(LogReg_recall, 6), round(LogReg_f1, 6)]
]

# Print the table
headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
print("\nModel Performance Comparison:")
print(tabulate(model_comparison, headers=headers, tablefmt="pretty"))
