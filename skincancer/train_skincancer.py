# Using this dataset:hmnist_28_28_RGB.csv
# Remove the first line (feature names)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Resizing
from tensorflow.keras.utils import to_categorical
import numpy as np

dataset = np.loadtxt("hmnist_28_28_RGB_preprocessed.csv", delimiter=",")

num_examples = dataset.shape[0]
num_features = dataset.shape[1] - 1
num_classes = 7

X = dataset[:, :-1]
y = dataset[:, -1]

X = X.reshape(num_examples, 28, 28, 3)
# Assuming y is the label array with integers from 0 to 6, we need to one-hot encode it
y_one_hot = to_categorical(y, num_classes=num_classes)

# Define the MobileNetV2 model without specifying the input shape
base_model = MobileNetV2(include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Build the model
model = models.Sequential([
    Resizing(32, 32, input_shape=(28, 28, 3)),  # Resizing layer to 32x32
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Training the model
model.fit(X, y_one_hot, epochs=10, batch_size=32, validation_split=0.2)

