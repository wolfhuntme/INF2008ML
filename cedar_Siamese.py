import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# This Model uses Siamese Network to compare two images and predict if they are similar or not (Deep Learning Model)
# The model uses a CNN to extract features from the images and then computes the difference between the two encodings
# The final layer is a Dense layer with a sigmoid activation to predict the similarity


# ========== STEP 1: Load & Preprocess the Dataset ==========
genuine_path = r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\small_org"  # Update this path
forged_path = r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\small_forg"   # Update this path
IMG_SIZE = (150, 150)

def load_images(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            img = load_img(file_path, target_size=IMG_SIZE)
            img = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(label)  # 1 for genuine, 0 for forged
    return np.array(images), np.array(labels)

# Load images for each class
genuine_images, genuine_labels = load_images(genuine_path, label=1)
forged_images, forged_labels = load_images(forged_path, label=0)

# Combine the two classes
X = np.concatenate([genuine_images, forged_images])
y = np.concatenate([genuine_labels, forged_labels])

print("Total Genuine Signatures:", len(genuine_images))
print("Total Forged Signatures:", len(forged_images))
print("Dataset Shape:", X.shape, "Labels Shape:", y.shape)

# ========== STEP 2: Three-Way Split ==========
# First, split off the test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Then, split the remaining data into training (60% overall) and development (20% overall)
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print("Training set size:", X_train.shape[0])
print("Development set size:", X_dev.shape[0])
print("Test set size:", X_test.shape[0])

# ========== STEP 3: Dynamic Pair Generation for Training ==========
def pair_generator(images, labels, batch_size):
    # Separate images by class
    genuine = [img for img, lbl in zip(images, labels) if lbl == 1]
    forged  = [img for img, lbl in zip(images, labels) if lbl == 0]
    
    while True:
        batch_pairs = []
        batch_labels = []
        for _ in range(batch_size):
            # Randomly choose to generate a positive or negative pair
            if random.random() < 0.5:
                # Positive pair: select two images from the same class
                if random.random() < 0.5:
                    if len(genuine) >= 2:
                        pair = random.sample(genuine, 2)
                        label = 1
                    else:
                        continue
                else:
                    if len(forged) >= 2:
                        pair = random.sample(forged, 2)
                        label = 1
                    else:
                        continue
            else:
                # Negative pair: select one from each class
                if genuine and forged:
                    pair = [random.choice(genuine), random.choice(forged)]
                    label = 0
                else:
                    continue
            batch_pairs.append(pair)
            batch_labels.append(label)
        
        batch_pairs = np.array(batch_pairs)
        batch_labels = np.array(batch_labels)
        # Yield as a tuple rather than a list
        yield (batch_pairs[:, 0], batch_pairs[:, 1]), batch_labels

batch_size = 32
train_gen = pair_generator(X_train, y_train, batch_size)

# Precompute pairs for the development and test sets for evaluation
def create_pairs(images, labels):
    pairs = []
    pair_labels = []
    num_samples = len(images)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if labels[i] == labels[j]:
                pairs.append([images[i], images[j]])
                pair_labels.append(1)
            else:
                pairs.append([images[i], images[j]])
                pair_labels.append(0)
    return np.array(pairs), np.array(pair_labels)

dev_pairs, dev_pair_labels = create_pairs(X_dev, y_dev)
test_pairs, test_pair_labels = create_pairs(X_test, y_test)
print("Development pairs shape:", dev_pairs.shape, "Development labels shape:", dev_pair_labels.shape)
print("Test pairs shape:", test_pairs.shape, "Test labels shape:", test_pair_labels.shape)

# ========== STEP 4: Build the Siamese Network ==========
def build_siamese_network(input_shape):
    input_layer = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    return Model(input_layer, x)

input_shape = (150, 150, 3)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# Shared CNN for both inputs
siamese_cnn = build_siamese_network(input_shape)
encoded_a = siamese_cnn(input_a)
encoded_b = siamese_cnn(input_b)

# Lambda layer to compute the absolute difference between encodings
distance = layers.Lambda(
    lambda tensors: tf.abs(tensors[0] - tensors[1]),
    output_shape=lambda input_shape: input_shape[0]
)([encoded_a, encoded_b])

# Final classification layer
prediction = layers.Dense(1, activation='sigmoid')(distance)
model = Model(inputs=[input_a, input_b], outputs=prediction)
model.compile(optimizer=Adam(0.0001), loss=BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# ========== STEP 5: Train the Model with Early Stopping ==========
# Use the development set to monitor validation performance
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
steps_per_epoch = 50  # Adjust this based on your data and training preference

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=([dev_pairs[:, 0], dev_pairs[:, 1]], dev_pair_labels),
    epochs=30,  # Set high; early stopping will end training if no improvement is seen
    callbacks=[early_stop]
)

# ========== STEP 6: Plot Training & Development Accuracy ==========
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Development Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ========== STEP 7: Evaluate the Model on Test Data ==========
test_loss, test_acc = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_pair_labels)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# (Optional) Save the trained model for later use.
model.save('siamese_signature_model.h5')

# ========== STEP 8: Inference with the Trained Model ==========
def predict_pair(image1_path, image2_path):
    img1 = load_img(image1_path, target_size=IMG_SIZE)
    img1 = img_to_array(img1) / 255.0
    img2 = load_img(image2_path, target_size=IMG_SIZE)
    img2 = img_to_array(img2) / 255.0
    
    prediction = model.predict([np.array([img1]), np.array([img2])])
    return "Genuine" if prediction > 0.5 else "Forged"

# Example usage (update the paths accordingly):
print(predict_pair(
    r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\full_org\original_22_14.png", 
    r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\full_org\original_22_15.png"))

print(predict_pair(
    r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\full_org\original_30_14.png", 
    r"C:\Users\Vyse\Documents\GitHub\Machine-Learning\signatures_cedar\full_forg\forgeries_30_15.png"))
