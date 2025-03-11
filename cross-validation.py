import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ========== PARAMETERS ==========
IMG_SIZE = (150, 150)

# HOG parameters (adjust to your preference)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'

def load_images_and_extract_features(folder, label):
    """Load images from `folder` and extract HOG features. Return feature matrix and labels."""
    features = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip unreadable images
            # Resize to a consistent size
            img = cv2.resize(img, IMG_SIZE)
            # Extract HOG features
            hog_features = hog(
                img,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm=block_norm
            )
            features.append(hog_features)
            labels.append(label)
    return np.array(features), np.array(labels)

# ========== PATHS ==========
genuine_path = r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\full_org"  # Update path
forged_path  = r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\full_forg" # Update path

# ========== LOAD DATA & EXTRACT FEATURES ==========
X_genuine, y_genuine = load_images_and_extract_features(genuine_path, label=1)
X_forged,  y_forged  = load_images_and_extract_features(forged_path, label=0)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

# Combine
X = np.concatenate([X_genuine, X_forged])
y = np.concatenate([y_genuine, y_forged])

print("Combined feature matrix shape:", X.shape)
print("Combined labels shape:", y.shape)

# ========== K-FOLD CROSS-VALIDATION ==========
k = 5  # Number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []
reports = []  # to store classification reports if desired

fold_idx = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train an SVM (linear kernel here; you can experiment with RBF, etc.)
    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Predict on the test fold
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    # Optionally collect classification reports
    rep = classification_report(y_test, y_pred, output_dict=True)
    reports.append(rep)
    
    print(f"Fold {fold_idx} Accuracy: {acc * 100:.2f}%")
    fold_idx += 1

# Print overall results
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"\n=== {k}-Fold Cross-Validation Results ===")
print(f"Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

# (Optional) Aggregate classification reports (averaging precision, recall, f1)
# For a more rigorous approach, see how each metric is aggregated across folds.
# Example:
avg_precision_0 = np.mean([r['0']['precision'] for r in reports])
avg_recall_0    = np.mean([r['0']['recall'] for r in reports])
avg_f1_0        = np.mean([r['0']['f1-score'] for r in reports])

avg_precision_1 = np.mean([r['1']['precision'] for r in reports])
avg_recall_1    = np.mean([r['1']['recall'] for r in reports])
avg_f1_1        = np.mean([r['1']['f1-score'] for r in reports])

print(f"\nClass 0 (forged) Avg Precision: {avg_precision_0:.2f}, Recall: {avg_recall_0:.2f}, F1: {avg_f1_0:.2f}")
print(f"Class 1 (genuine) Avg Precision: {avg_precision_1:.2f}, Recall: {avg_recall_1:.2f}, F1: {avg_f1_1:.2f}")
