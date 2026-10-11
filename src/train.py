#!/usr/bin/env python3
"""
Step 4: CNN Training Script
Trains lightweight custom CNN with EarlyStopping, ModelCheckpoint, class weighting,
and plots training/validation curves to results/metrics/training_curves.png.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Ensure src directory is in sys.path regardless of execution CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from model import build_custom_cnn
except ImportError:
    from src.model import build_custom_cnn

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_split_images(manifest_df, split_name):
    """
    Loads images and labels for a specific split from manifest.
    Normalizes pixel values to [0, 1] float32.
    """
    split_df = manifest_df[manifest_df['split'] == split_name].reset_index(drop=True)
    images = []
    labels = []
    
    print(f"Loading {len(split_df)} images for split: '{split_name}'...")
    for idx, row in split_df.iterrows():
        img_path = row['file_path']
        # Load as RGB
        pil_img = Image.open(img_path).convert('RGB')
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        images.append(img_arr)
        labels.append(row['class'])
        
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y, split_df


def train_cnn(manifest_path="data/processed/encoded_images_manifest.csv",
              model_save_path="models/cnn_asd_model.h5",
              metrics_dir="results/metrics",
              batch_size=16,
              epochs=40):
    """
    Main training routine:
      - Loads train images from manifest
      - Stratified train/val split (80/20) on train split
      - Computes balanced class weights
      - Compiles and trains custom CNN with callbacks
      - Saves model and plots loss/accuracy curves
    """
    print("=" * 60)
    print("STEP 4: CNN MODEL TRAINING")
    print("=" * 60)
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run cross_modal_encoding.py first.")
        
    manifest_df = pd.read_csv(manifest_path)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Load train split data
    X_train_full, y_train_full, _ = load_split_images(manifest_df, "train")
    print(f"Full training data shape: X={X_train_full.shape}, y={y_train_full.shape}")

    # 2. Stratified train/validation split (80% train, 20% validation)
    # The test set remains strictly held-out and untouched for final evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=RANDOM_SEED
    )
    print(f"Train set: {X_train.shape[0]} samples (Pos: {int(y_train.sum())}, Neg: {int(len(y_train) - y_train.sum())})")
    print(f"Val set:   {X_val.shape[0]} samples (Pos: {int(y_val.sum())}, Neg: {int(len(y_val) - y_val.sum())})")

    # 3. Compute Class Weights for Imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Class weights applied: {class_weight_dict}")

    # Note on Data Augmentation:
    # Because each row/column in the encoded matrix corresponds strictly to a specific feature index
    # (e.g. Row 0 = A1, Row 10 = Age), spatial transformations like rotation or zooming distort the
    # fixed coordinate grid semantics. Hence, no destructive spatial transformations are used.
    print("\nNote: Destructive spatial augmentations (rotation/shear) omitted to preserve coordinate semantics.")

    # 4. Build Model
    model = build_custom_cnn(input_shape=(224, 224, 3), learning_rate=3e-4)

    # 5. Callbacks: EarlyStopping and ModelCheckpoint
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 6. Fit Model
    print(f"\nStarting model training for up to {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Also save in .keras format for native Keras 3 compatibility
    keras_save_path = model_save_path.replace(".h5", ".keras")
    model.save(keras_save_path)
    print(f"Model successfully saved to {model_save_path} and {keras_save_path}")

    # 7. Plot training and validation curves
    print("\nPlotting training and validation curves...")
    epochs_trained = len(history.history['loss'])
    epoch_range = range(1, epochs_trained + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epoch_range, history.history['loss'], 'b-o', label='Training Loss', markersize=4)
    axes[0].plot(epoch_range, history.history['val_loss'], 'r--s', label='Validation Loss', markersize=4)
    axes[0].set_title('Training and Validation Loss', fontsize=12, weight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Binary Cross-Entropy Loss', fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy curve
    axes[1].plot(epoch_range, history.history['accuracy'], 'b-o', label='Training Accuracy', markersize=4)
    axes[1].plot(epoch_range, history.history['val_accuracy'], 'r--s', label='Validation Accuracy', markersize=4)
    axes[1].set_title('Training and Validation Accuracy', fontsize=12, weight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    training_curves_path = os.path.join(metrics_dir, "training_curves.png")
    plt.savefig(training_curves_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {training_curves_path}")

    # Print summary of best metrics
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n--- TRAINING SUMMARY ---")
    print(f"Total Epochs Trained: {epochs_trained}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    return model, history


if __name__ == "__main__":
    train_cnn()
