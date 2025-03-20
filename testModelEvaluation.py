import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tabulate import tabulate
import time
from memory_profiler import memory_usage

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

def evaluate_model(model_file, model_name, X_data, y_true):
    """Load a model, compute file size, inference time, memory overhead, and performance metrics."""
    # File size in KB
    file_size = os.path.getsize(model_file) / 1024  # in KB
    
    # Load the model
    model = joblib.load(model_file)
    print(f"{model_name} model loaded.")
    
    # Measure inference time
    start_time = time.time()
    predictions = model.predict(X_data)
    inference_time = time.time() - start_time

    # Measure memory overhead during prediction using memory_usage.
    # This runs the prediction function separately.
    mem_usage = memory_usage((model.predict, (X_data,)), interval=0.01)
    memory_overhead = max(mem_usage) - min(mem_usage)
    
    # Compute performance metrics
    accuracy = accuracy_score(y_true, predictions)
    report = classification_report(y_true, predictions, output_dict=True)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    return [model_name, round(file_size, 2), round(inference_time, 4),
            round(memory_overhead, 4), round(accuracy, 6), round(precision, 6),
            round(recall, 6), round(f1, 6)]

# ========== PATHS FOR UNSEEN DATA ==========
genuine_unseen_folder = r"C:\Users\Vyse\Documents\GitHub\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_org"   # Update this path
forged_unseen_folder  = r"C:\Users\Vyse\Documents\GitHub\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_forg"    # Update this path

if __name__ == '__main__':
    # Load unseen data and extract features
    X_unseen, y_unseen = load_unseen_data(genuine_unseen_folder, forged_unseen_folder)
    print("Unseen data samples:", X_unseen.shape)

    # Evaluate models on unseen data
    models_info = [
        ("adaboost_model.pkl", "AdaBoost"),
        ("random_forest_model.pkl", "Random Forest"),
        ("knn_model.pkl", "KNN"),
        ("svm_model.pkl", "SVM"),
        ("writer_independent_logreg_model.pkl", "LogReg")
    ]

    results = []
    for model_file, model_name in models_info:
        metrics = evaluate_model(model_file, model_name, X_unseen, y_unseen)
        results.append(metrics)

    headers = ["Model", "File Size (KB)", "Inference Time (s)", "Memory Overhead (MiB)",
               "Accuracy", "Precision", "Recall", "F1-Score"]
    print("\nModel Performance Comparison:")
    print(tabulate(results, headers=headers, tablefmt="pretty"))

