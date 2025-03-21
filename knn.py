import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, make_scorer, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib  # For model persistence
import seaborn as sns  # For improved visualization of confusion matrix

# HOG Parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
IMG_SIZE = (150, 150)
# Set this to True if you want to use PCA for dimensionality reduction
USE_PCA = True
PCA_COMPONENTS = 100  # Adjust based on your feature dimension and variance explained

# Function to extract HOG features from an image
def load_images_and_extract_features(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Unable to load image {file_path}")
                continue
            try:
                img = cv2.resize(img, IMG_SIZE)
            except Exception as e:
                print(f"Error resizing {file_path}: {e}")
                continue
            hog_features = hog(img,
                               orientations=orientations,
                               pixels_per_cell=pixels_per_cell,
                               cells_per_block=cells_per_block,
                               block_norm='L2-Hys')
            features.append(hog_features)
            labels.append(label)
    return np.array(features), np.array(labels)

# Dataset path for training the model
genuine_path = r"C:\Users\khooa\Documents\GitHub\INF2008ML\signatures_cedar\full_org"
forged_path  = r"C:\Users\khooa\Documents\GitHub\INF2008ML\signatures_cedar\full_forg"

X_genuine, y_genuine = load_images_and_extract_features(genuine_path, label=1)
X_forged,  y_forged  = load_images_and_extract_features(forged_path, label=0)

print("Total Genuine Signatures:", len(X_genuine))
print("Total Forged Signatures:", len(X_forged))

X = np.concatenate([X_genuine, X_forged])
y = np.concatenate([y_genuine, y_forged])

print("Combined feature matrix shape:", X.shape)
print("Combined labels shape:", y.shape)

# Split data into training, testing and developement set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print("Training set size:", X_train.shape[0])
print("Development set size:", X_dev.shape[0])
print("Test set size:", X_test.shape[0])

# Custom scorer for GridSearchCV to optimize for recall of the forged class
def forged_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

forged_recall_scorer = make_scorer(forged_recall)

# Construct the pipeline
steps = [
    ('scaler', StandardScaler())
]

if USE_PCA:
    steps.append(('pca', PCA(n_components=PCA_COMPONENTS)))
    
steps.append(('knn', KNeighborsClassifier()))
pipeline = Pipeline(steps)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=forged_recall_scorer, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

# Use the best estimator from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate on Development Set
y_dev_pred = best_model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print("Development Set Accuracy (KNN): {:.2f}%".format(dev_accuracy * 100))
print("\nDevelopment Classification Report (KNN):")
print(classification_report(y_dev, y_dev_pred))

# Evaluate on Test Set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Set Accuracy (KNN): {:.2f}%".format(test_accuracy * 100))
print("\nTest Classification Report (KNN):")
print(classification_report(y_test, y_test_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Forged", "Genuine"], yticklabels=["Forged", "Genuine"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix on Test Set")
plt.show()

joblib.dump(best_model, "knn_model.pkl")
print("Optimized KNN model saved as knn_model_improved.pkl")

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
                                              orientations=orientations,
                                              pixels_per_cell=(8,8),
                                              cells_per_block=(2,2),
                                              block_norm='L2-Hys',
                                              visualize=True)
    forged_features, forged_hog_image = hog(forged_img,
                                            orientations=orientations,
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

compare_hog(r"C:\Users\khooa\Documents\GitHub\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_org\original_41_1.png", 
            r"C:\Users\khooa\Documents\GitHub\INF2008ML\signatures_cedar\unseen_data_for_testing\unseen_forg\forgeries_41_1.png")
