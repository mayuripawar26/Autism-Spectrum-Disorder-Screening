#!/usr/bin/env python3
"""
Step 3: Cross-Modal Feature Encoding
Converts tabular behavioral and demographic features into 224x224 RGB images:
  - Channel R: Outer product matrix (pairwise interaction)
  - Channel G: Absolute difference matrix (contrast)
  - Channel B: Train Pearson correlation elementwise * interaction, rescaled [-1,1] -> [0,1]
Upsampled using NEAREST NEIGHBOR interpolation to 224x224.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def encode_feature_vector(feature_vector, global_corr_matrix):
    """
    Encodes an n-dimensional feature vector into an n x n x 3 RGB array.
      - Channel R: outer product matrix, M[i][j] = f_i * f_j
      - Channel G: absolute difference matrix, M[i][j] = |f_i - f_j|
      - Channel B: global_corr_matrix * M[i][j], rescaled [-1,1] to [0,1]
    """
    f = np.array(feature_vector, dtype=np.float32)
    n = len(f)
    
    # Channel R: Outer product matrix (pairwise interaction)
    channel_r = np.outer(f, f)
    
    # Channel G: Absolute difference matrix (contrast)
    # Using broadcasting: |f_i - f_j|
    channel_g = np.abs(f[:, np.newaxis] - f[np.newaxis, :])
    
    # Channel B: Pearson correlation * interaction, rescaled from [-1, 1] to [0, 1]
    interaction = channel_r
    weighted_corr = global_corr_matrix * interaction
    channel_b = (weighted_corr + 1.0) / 2.0
    
    # Ensure values are strictly bounded in [0, 1]
    channel_r = np.clip(channel_r, 0.0, 1.0)
    channel_g = np.clip(channel_g, 0.0, 1.0)
    channel_b = np.clip(channel_b, 0.0, 1.0)
    
    # Stack into RGB array: shape (n, n, 3)
    rgb_matrix = np.stack([channel_r, channel_g, channel_b], axis=-1)
    
    return rgb_matrix


def resize_and_save_image(rgb_matrix, output_path, target_size=(224, 224)):
    """
    Upsamples matrix to target_size using NEAREST NEIGHBOR interpolation
    and saves as PNG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert float [0, 1] to uint8 [0, 255]
    img_uint8 = np.clip(rgb_matrix * 255.0, 0, 255).astype(np.uint8)
    
    # Upsample using NEAREST NEIGHBOR to preserve block boundaries
    img_resized = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Save using PIL Image (preserves RGB channels without OpenCV BGR inversion)
    pil_img = Image.fromarray(img_resized, mode='RGB')
    pil_img.save(output_path, format='PNG')
    return img_resized


def encode_dataset(cleaned_csv_path="data/processed/cleaned_data.csv",
                   output_base_dir="data/processed/encoded_images",
                   results_dir="results"):
    """
    Main function to run 80/20 stratified split, compute global correlation matrix
    on train split only, encode all samples into 224x224 RGB images, and create
    sanity-check visualization.
    """
    print("=" * 60)
    print("STEP 3: CROSS-MODAL FEATURE ENCODING")
    print("=" * 60)
    
    if not os.path.exists(cleaned_csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_csv_path}. Run data_preprocessing.py first.")
        
    df = pd.read_csv(cleaned_csv_path)
    target_col = 'Class/ASD'
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)
    print(f"Loaded {len(df)} samples with {n_features} features: {feature_cols}")

    # 1. 80/20 Stratified train/test split BEFORE encoding to prevent data leakage
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    
    print(f"Train split size: {len(X_train)} samples ({y_train.sum()} High Risk, {len(y_train) - y_train.sum()} Low Risk)")
    print(f"Test split size:  {len(X_test)} samples ({y_test.sum()} High Risk, {len(y_test) - y_test.sum()} Low Risk)")

    # Save train/test feature splits for full reproducibility
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv("data/processed/train_data.csv", index=True)
    test_df.to_csv("data/processed/test_data.csv", index=True)

    # 2. Compute Pearson correlation matrix ONLY on the training split
    print("\nComputing Pearson correlation matrix on training split only...")
    corr_matrix_df = X_train.corr(method='pearson')
    global_corr_matrix = np.nan_to_num(corr_matrix_df.to_numpy(dtype=np.float32), nan=0.0)
    
    # Save correlation matrix
    np.save("data/processed/train_correlation_matrix.npy", global_corr_matrix)
    print(f"Global correlation matrix shape: {global_corr_matrix.shape}")
    print(f"Correlation min: {global_corr_matrix.min():.4f}, max: {global_corr_matrix.max():.4f}, mean: {global_corr_matrix.mean():.4f}")

    # Save feature names mapping for Grad-CAM interpretation
    feature_mapping = {i: col for i, col in enumerate(feature_cols)}
    with open("data/processed/feature_mapping.json", "w") as f:
        json.dump(feature_mapping, f, indent=2)
    print(f"Saved feature index mapping to: data/processed/feature_mapping.json")

    # 3. Encode and save images for Train and Test splits
    splits = [
        ('train', X_train, y_train),
        ('test', X_test, y_test)
    ]
    
    manifest_records = []

    for split_name, X_split, y_split in splits:
        print(f"\nEncoding {split_name} split ({len(X_split)} images)...")
        for orig_idx, (feat_vals, label) in zip(X_split.index, zip(X_split.values, y_split.values)):
            class_folder = f"class_{int(label)}"
            filename = f"sample_{orig_idx}.png"
            output_path = os.path.join(output_base_dir, split_name, class_folder, filename)
            
            # Encode to RGB matrix
            rgb_matrix = encode_feature_vector(feat_vals, global_corr_matrix)
            
            # Upsample and save 224x224 image
            resize_and_save_image(rgb_matrix, output_path, target_size=(224, 224))
            
            manifest_records.append({
                "sample_idx": int(orig_idx),
                "split": split_name,
                "class": int(label),
                "file_path": output_path.replace("\\", "/")
            })

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv("data/processed/encoded_images_manifest.csv", index=False)
    print(f"\nAll images encoded and saved successfully! Manifest: data/processed/encoded_images_manifest.csv")

    # 4. Visual sanity check: 3 random images from each class
    print("\nGenerating visual sanity-check plot (3 random images per class)...")
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Visual Sanity-Check: Cross-Modal Encoded Images (224x224 RGB)", fontsize=14, weight='bold')

    classes = [0, 1]
    class_labels = {0: "Low Risk (Class 0)", 1: "High Risk (Class 1)"}

    for row_idx, cls in enumerate(classes):
        cls_manifest = manifest_df[manifest_df['class'] == cls]
        sampled_records = cls_manifest.sample(n=3, random_state=RANDOM_SEED).to_dict('records')
        
        for col_idx, rec in enumerate(sampled_records):
            img_path = rec['file_path']
            # Load with PIL to ensure RGB color space
            pil_img = Image.open(img_path)
            img_np = np.array(pil_img)
            
            ax = axes[row_idx, col_idx]
            ax.imshow(img_np)
            ax.set_title(f"{class_labels[cls]}\nSample #{rec['sample_idx']} ({rec['split']})", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    sanity_check_path = os.path.join(results_dir, "sample_encoded_images.png")
    plt.savefig(sanity_check_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Sanity check plot saved to: {sanity_check_path}")

    # Verify images are not blank or degenerate
    for rec in manifest_records[:10]:
        check_img = np.array(Image.open(rec['file_path']))
        if check_img.std() < 1e-3:
            raise ValueError(f"Warning: Image {rec['file_path']} appears degenerate (std={check_img.std()})!")
            
    print("Sanity verification passed: Encoded images show clear, non-degenerate cross-modal patterns.")
    return manifest_df


if __name__ == "__main__":
    encode_dataset()
