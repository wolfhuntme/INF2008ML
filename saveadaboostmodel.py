import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
# ========== PARAMETERS ==========
IMG_SIZE = (150, 150)
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
genuine_path = r"C:\Users\aqmal\OneDrive\Desktop\INF2008ML-main\signatures_cedar\full_org"  # Update path
forged_path  = r"C:\Users\aqmal\OneDrive\Desktop\INF2008ML-main\signatures_cedar\full_forg" # Update path

# ========== LOAD DATA & EXTRACT FEATURES ==========
X_genuine, y_genuine = load_images_and_extract_features(genuine_path, label=1)
X_forged,  y_forged  = load_images_and_extract_features(forged_path, label=0)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

# Combine
X = np.concatenate([X_genuine, X_forged])
y = np.concatenate([y_genuine, y_forged])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the AdaBoost model
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Train the AdaBoost model
ada_model.fit(X_train, y_train)

# Save the trained AdaBoost model
joblib.dump(ada_model, "adaboost_model.pkl")