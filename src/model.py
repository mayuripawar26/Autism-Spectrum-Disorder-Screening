#!/usr/bin/env python3
"""
Step 4: Lightweight Custom CNN Architecture for ASD Screening
Designed for 224x224x3 cross-modal encoded feature matrices.
Controls parameter count via GlobalAveragePooling2D to prevent overfitting
on small sample regimes (~700 samples).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(input_shape=(224, 224, 3), learning_rate=3e-4):
    """
    Constructs a 3-block convolutional neural network with BatchNorm,
    GlobalAveragePooling2D, Dense, Dropout, and Sigmoid output.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Block 3: 128 filters (Target layer for Grad-CAM)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_pred")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="ASD_CrossModal_CNN")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    return model


if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
