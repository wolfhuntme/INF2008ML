###########################################################
# COMPLETE CODE: SUBJECT-BASED SPLIT + SIAMESE + SVM
###########################################################

import os
import re
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# ----- PARAMETERS -----
IMG_SIZE = (150, 150)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'

###########################################################
# 1. Parse filenames to get subject IDs
###########################################################

def parse_subject_id(filename):
    """
    Example parser for filenames like:
      'original_1_4' or 'forgeries_1_4'
    which indicates subject = 1.
    
    If your naming is different (e.g. forgeries1_4),
    adjust this regex or logic accordingly.
    """
    # We'll try a pattern that captures the first integer after an underscore.
    match = re.search(r'_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        # If we can't parse, return None or handle accordingly
        return None

###########################################################
# 2. Load images by subject
###########################################################

def load_images_by_subject(folder, label):
    """
    Returns a dict: subject_id -> list of (img_array, label).
    """
    subject_dict = {}
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            subj_id = parse_subject_id(filename)
            if subj_id is None:
                continue  # skip if we can't parse a subject
            try:
                img = load_img(file_path, target_size=IMG_SIZE)
                img = img_to_array(img) / 255.0  # Normalize
                if subj_id not in subject_dict:
                    subject_dict[subj_id] = []
                subject_dict[subj_id].append((img, label))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return subject_dict

###########################################################
# 3. Combine "genuine" & "forged" images into one dict
###########################################################

# Adjust these paths to your dataset structure:
genuine_path = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\small_org"
forged_path  = r"C:\Users\xavie\Desktop\School\INF2008ML\signatures_cedar\small_forg"

genuine_subjects = load_images_by_subject(genuine_path, label=1)
forged_subjects  = load_images_by_subject(forged_path, label=0)

all_subject_dict = {}

# Merge genuine_subjects
for subj_id in genuine_subjects:
    if subj_id not in all_subject_dict:
        all_subject_dict[subj_id] = []
    all_subject_dict[subj_id].extend(genuine_subjects[subj_id])

# Merge forged_subjects
for subj_id in forged_subjects:
    if subj_id not in all_subject_dict:
        all_subject_dict[subj_id] = []
    all_subject_dict[subj_id].extend(forged_subjects[subj_id])

# Print how many subjects total
all_subject_ids = list(all_subject_dict.keys())
print(f"Total subjects: {len(all_subject_ids)}")

###########################################################
# 4. Split by subjects, not images
###########################################################

train_val_subj, test_subj = train_test_split(all_subject_ids, test_size=0.2, random_state=42)
print(f"Training+Validation subjects: {train_val_subj}")
print(f"Test subjects: {test_subj}")

# Convert subject-based dictionary to arrays for train+val, test
train_val_imgs, train_val_labels = [], []
for subj in train_val_subj:
    for (img, lbl) in all_subject_dict[subj]:
        train_val_imgs.append(img)
        train_val_labels.append(lbl)

train_val_imgs   = np.array(train_val_imgs)
train_val_labels = np.array(train_val_labels)

test_imgs, test_labels = [], []
for subj in test_subj:
    for (img, lbl) in all_subject_dict[subj]:
        test_imgs.append(img)
        test_labels.append(lbl)

test_imgs   = np.array(test_imgs)
test_labels = np.array(test_labels)

# Next split train_val_imgs -> train(80% of them?), val(20% of them?). 
# Example: 80% train, 20% validation. Adjust to your preference.
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    train_val_imgs, train_val_labels, test_size=0.2, random_state=42)

print(f"Train set images: {train_imgs.shape[0]}")
print(f"Validation set images: {val_imgs.shape[0]}")
print(f"Test set images: {test_imgs.shape[0]}")

###########################################################
# 5. Build pairs for train, val, and test
###########################################################

def create_pairs(images, labels):
    """
    Creates all unique pairs from a set of images.
    pair_labels = 1 if same class, else 0.
    WARNING: This can be huge if you have many images.
    Consider random sampling if memory is an issue.
    """
    pairs = []
    pair_labels = []
    num_samples = len(images)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            pairs.append([images[i], images[j]])
            pair_labels.append(1 if labels[i] == labels[j] else 0)
    return np.array(pairs), np.array(pair_labels)

train_pairs, train_pair_labels = create_pairs(train_imgs, train_labels)
val_pairs,   val_pair_labels   = create_pairs(val_imgs,   val_labels)
test_pairs,  test_pair_labels  = create_pairs(test_imgs,  test_labels)

print(f"Train pairs: {train_pairs.shape}, {train_pair_labels.shape}")
print(f"Val pairs:   {val_pairs.shape},   {val_pair_labels.shape}")
print(f"Test pairs:  {test_pairs.shape},  {test_pair_labels.shape}")

###########################################################
# 6. Build and Train the Siamese Network
###########################################################

def build_siamese_network(input_shape):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    return Model(inp, x)

IMG_SIZE = (150, 150)
input_shape = IMG_SIZE + (3,)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

shared_cnn = build_siamese_network(input_shape)
encoded_a = shared_cnn(input_a)
encoded_b = shared_cnn(input_b)

# L1 distance
L1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_a, encoded_b])
prediction = layers.Dense(1, activation='sigmoid')(L1_distance)

siamese_model = Model(inputs=[input_a, input_b], outputs=prediction)
siamese_model.compile(optimizer=Adam(0.0001), loss=BinaryCrossentropy(), metrics=['accuracy'])
siamese_model.summary()

# Prepare pairs for training
train_a = train_pairs[:, 0]
train_b = train_pairs[:, 1]
val_a   = val_pairs[:, 0]
val_b   = val_pairs[:, 1]

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_siamese = siamese_model.fit(
    [train_a, train_b], train_pair_labels,
    validation_data=([val_a, val_b], val_pair_labels),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

# Save the trained model
siamese_model.save("siamese_signature_model.h5")
print("Siamese model saved as siamese_signature_model.h5")

###########################################################
# 7. HOG + SVM
###########################################################

def extract_hog(image):
    """
    Convert [0,1]-range image to 8-bit grayscale, then extract HOG features.
    """
    image_uint8 = (image * 255).astype('uint8')
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm=block_norm)
    return hog_features

def create_difference_features(pair_array):
    """
    For each pair, compute absolute difference of HOG features.
    """
    diff_features = []
    for pair in pair_array:
        feat1 = extract_hog(pair[0])
        feat2 = extract_hog(pair[1])
        diff_features.append(np.abs(feat1 - feat2))
    return np.array(diff_features)

# Build training, val, and test sets for SVM
svm_train_features = create_difference_features(train_pairs)
svm_val_features   = create_difference_features(val_pairs)
svm_test_features  = create_difference_features(test_pairs)

svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
svm_classifier.fit(svm_train_features, train_pair_labels)

svm_val_pred  = svm_classifier.predict(svm_val_features)
val_acc_svm   = accuracy_score(val_pair_labels, svm_val_pred)
print(f"SVM Validation Accuracy: {val_acc_svm*100:.2f}%")

svm_test_pred = svm_classifier.predict(svm_test_features)
test_acc_svm  = accuracy_score(test_pair_labels, svm_test_pred)
print(f"SVM Test Accuracy: {test_acc_svm*100:.2f}%")

# Save the SVM model
joblib.dump(svm_classifier, 'svm_signature_model.pkl')
print("SVM saved as svm_signature_model.pkl")

###########################################################
# 8. Ensemble Prediction
###########################################################

def ensemble_prediction(image_pair, weight=0.7):
    """
    Weighted average of siamese_pred and svm_pred.
    """
    siamese_pred = siamese_model.predict([
        np.array([image_pair[0]]),
        np.array([image_pair[1]])
    ])[0][0]
    
    feat1 = extract_hog(image_pair[0])
    feat2 = extract_hog(image_pair[1])
    diff = np.abs(feat1 - feat2).reshape(1, -1)
    svm_pred_prob = svm_classifier.predict_proba(diff)[0][1]

    combined_score = weight * siamese_pred + (1 - weight) * svm_pred_prob
    final_label = 1 if combined_score > 0.5 else 0
    return final_label, combined_score

# Evaluate ensemble on the test set
test_a = test_pairs[:, 0]
test_b = test_pairs[:, 1]

ensemble_preds = []
for i in range(len(test_pairs)):
    label, _ = ensemble_prediction([test_a[i], test_b[i]])
    ensemble_preds.append(label)

ensemble_accuracy = accuracy_score(test_pair_labels, ensemble_preds)
print(f"Ensemble Test Accuracy: {ensemble_accuracy*100:.2f}%")

###########################################################
# 9. Example Inference on Single Pair
###########################################################

# Suppose you have two images:
test_image1_path = r"C:\Users\xavie\Desktop\School\INF2008ML\Dataset_Signature_Final\Dataset\dataset1\forge\02100001.png" # e.g. a forged signature
test_image2_path = r"C:\Users\xavie\Desktop\School\INF2008ML\Dataset_Signature_Final\Dataset\dataset1\real\00100001.png"  # e.g. a genuine signature

def load_and_preprocess_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return img

img1 = load_and_preprocess_image(test_image1_path)
img2 = load_and_preprocess_image(test_image2_path)
ens_label, ens_score = ensemble_prediction([img1, img2], weight=0.7)
print("Single Inference Ensemble Prediction:", "Genuine" if ens_label == 1 else "Forged", f"(Score={ens_score:.2f})")

###########################################################
# NOTES:
# - This code does a subject-based split to avoid overfitting.
# - If you have multiple persons with many images each,
#   you reduce the risk that person X's signature is both in train and test.
# - If memory is an issue with create_pairs (since it can create many pairs),
#   consider random sampling or a generator approach.
# - Check your final performance on the test set to ensure no leakage.
###########################################################
