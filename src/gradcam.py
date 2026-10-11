#!/usr/bin/env python3
"""
Step 5: Grad-CAM Implementation for Explainable ASD Screening
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
for convolutional layers, overlaying explanations on the 224x224 encoded images.
Maps spatial activations back to original behavioral and demographic features.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf

def load_model_for_gradcam(model_path="models/cnn_asd_model.keras"):
    """Loads trained model, checking both .keras and .h5 formats."""
    if not os.path.exists(model_path):
        alt_path = model_path.replace(".keras", ".h5")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}")
            
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv3", pred_index=None):
    """
    Computes Grad-CAM heatmap for a single input image array of shape (1, 224, 224, 3).
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        prob = predictions[0][0]
        
        # If pred_index is None, explain the predicted class
        if pred_index is None:
            pred_class = 1 if prob >= 0.5 else 0
        else:
            pred_class = pred_index
            
        # Target score for class
        loss = predictions[:, 0] if pred_class == 1 else (1.0 - predictions[:, 0])

    # Gradient of target score with respect to feature maps of last conv layer
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients (importance weight alpha_k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of feature maps
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Pass through ReLU to keep only positive influence
    cam = tf.maximum(cam, 0)

    # Normalize heatmap between 0 and 1
    max_val = tf.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val
    heatmap = cam.numpy()

    return heatmap, prob.numpy(), pred_class


def map_heatmap_to_features(heatmap, feature_mapping, grid_size=14):
    """
    Maps 2D activation heatmap back to tabular feature interaction grid (14x14).
    Returns top activated feature pairs.
    """
    # Downsample or pool heatmap to grid_size x grid_size
    grid_heatmap = cv2.resize(heatmap, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    
    # Find top feature pairs
    top_indices = np.argsort(grid_heatmap.flatten())[::-1]
    
    top_pairs = []
    seen = set()
    for idx in top_indices:
        row = idx // grid_size
        col = idx % grid_size
        pair = tuple(sorted((row, col)))
        if pair not in seen:
            seen.add(pair)
            top_pairs.append({
                "feature_i": feature_mapping.get(str(row), feature_mapping.get(row, f"F{row}")),
                "feature_j": feature_mapping.get(str(col), feature_mapping.get(col, f"F{col}")),
                "activation": float(grid_heatmap[row, col])
            })
        if len(top_pairs) >= 5:
            break
            
    return grid_heatmap, top_pairs


def overlay_heatmap_on_image(img_arr, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlays resized colormap heatmap onto original image (values in [0, 1]).
    """
    # Resize heatmap to match image size (224, 224)
    heatmap_resized = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Superimpose
    superimposed = (1.0 - alpha) * img_arr + alpha * color_heatmap
    superimposed = np.clip(superimposed, 0.0, 1.0)
    
    return superimposed, color_heatmap


def run_gradcam_analysis(manifest_path="data/processed/encoded_images_manifest.csv",
                         model_path="models/cnn_asd_model.keras",
                         feature_mapping_path="data/processed/feature_mapping.json",
                         output_dir="results/gradcam_heatmaps"):
    """
    Generates Grad-CAM heatmaps for at least 5 correctly classified and 2 misclassified
    (or lowest-margin boundary) test samples, and saves visualizations.
    """
    print("=" * 60)
    print("STEP 5: GRAD-CAM EXPLAINABILITY")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    model = load_model_for_gradcam(model_path)
    
    with open(feature_mapping_path, "r") as f:
        feature_mapping = json.load(f)

    manifest_df = pd.read_csv(manifest_path)
    test_df = manifest_df[manifest_df['split'] == 'test'].reset_index(drop=True)
    print(f"Total held-out test samples: {len(test_df)}")

    # Load all test images and get model predictions
    test_imgs = []
    for p in test_df['file_path']:
        pil_img = Image.open(p).convert('RGB')
        test_imgs.append(np.array(pil_img, dtype=np.float32) / 255.0)
    test_imgs = np.array(test_imgs, dtype=np.float32)

    preds = model.predict(test_imgs, verbose=0)[:, 0]
    test_df['pred_prob'] = preds
    test_df['pred_class'] = (preds >= 0.5).astype(int)
    test_df['is_correct'] = (test_df['class'] == test_df['pred_class'])
    test_df['error_magnitude'] = np.abs(test_df['class'] - test_df['pred_prob'])

    # Test set performance summary
    test_acc = test_df['is_correct'].mean()
    misclassified_df = test_df[~test_df['is_correct']].sort_values(by='error_magnitude', ascending=False)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Number of misclassified test samples: {len(misclassified_df)}")

    # Select target samples for Grad-CAM
    # Need: at least 5 correctly classified samples and at least 2 misclassified
    correct_samples = []
    # Pick diverse correct samples (e.g. 3 High Risk class 1, 3 Low Risk class 0)
    correct_class1 = test_df[(test_df['is_correct']) & (test_df['class'] == 1)].sample(n=min(3, (test_df['class'] == 1).sum()), random_state=42)
    correct_class0 = test_df[(test_df['is_correct']) & (test_df['class'] == 0)].sample(n=min(3, (test_df['class'] == 0).sum()), random_state=42)
    correct_samples = pd.concat([correct_class1, correct_class0]).to_dict('records')

    # If misclassified samples in test < 2, supplement with highest uncertainty / boundary samples
    # or check the train split for any misclassifications
    misclassified_samples = misclassified_df.to_dict('records')
    if len(misclassified_samples) < 2:
        print("Note: Fewer than 2 misclassifications in test set due to high model accuracy.")
        print("Checking train/val split for additional misclassified or boundary edge-case samples...")
        train_df = manifest_df[manifest_df['split'] == 'train'].reset_index(drop=True)
        train_imgs = np.array([np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0 for p in train_df['file_path']])
        train_preds = model.predict(train_imgs, verbose=0)[:, 0]
        train_df['pred_prob'] = train_preds
        train_df['pred_class'] = (train_preds >= 0.5).astype(int)
        train_df['is_correct'] = (train_df['class'] == train_df['pred_class'])
        train_df['error_magnitude'] = np.abs(train_df['class'] - train_df['pred_prob'])
        train_mis = train_df[~train_df['is_correct']].sort_values(by='error_magnitude', ascending=False).to_dict('records')
        
        for item in train_mis:
            if len(misclassified_samples) >= 2:
                break
            misclassified_samples.append(item)
            
        # If still < 2, take the boundary test samples closest to 0.5 decision threshold
        if len(misclassified_samples) < 2:
            boundary_samples = test_df.sort_values(by='error_magnitude', ascending=False).to_dict('records')
            for b in boundary_samples:
                if b not in misclassified_samples:
                    misclassified_samples.append(b)
                if len(misclassified_samples) >= 2:
                    break

    selected_samples = []
    for s in correct_samples[:6]:
        s['category'] = 'Correct'
        selected_samples.append(s)
    for s in misclassified_samples[:2]:
        s['category'] = 'Misclassified / Edge Case' if not s['is_correct'] else 'Decision Boundary Edge Case'
        selected_samples.append(s)

    print(f"\nGenerating Grad-CAM for {len(selected_samples)} selected samples...")
    explanations_summary = []

    for idx, sample in enumerate(selected_samples):
        orig_idx = sample['sample_idx']
        true_cls = sample['class']
        pred_cls = sample['pred_class']
        prob = sample['pred_prob']
        category = sample['category']
        img_path = sample['file_path']

        img_pil = Image.open(img_path).convert('RGB')
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_arr, axis=0)

        # Compute Grad-CAM heatmap on 'conv3' layer
        heatmap, pred_prob, explained_class = make_gradcam_heatmap(
            img_tensor, model, last_conv_layer_name="conv3"
        )
        
        superimposed, color_heatmap = overlay_heatmap_on_image(img_arr, heatmap, alpha=0.5)
        grid_heatmap, top_pairs = map_heatmap_to_features(heatmap, feature_mapping, grid_size=14)

        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Original Image
        axes[0].imshow(img_arr)
        axes[0].set_title(f"Original Encoded Image\nSample #{orig_idx} ({category})", fontsize=11, weight='bold')
        axes[0].axis('off')

        # Panel 2: Grad-CAM Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f"Grad-CAM Heatmap ('conv3')\nExplaining Class {explained_class}", fontsize=11, weight='bold')
        axes[1].axis('off')

        # Panel 3: Superimposed Overlay
        axes[2].imshow(superimposed)
        axes[2].set_title(f"Superimposed Overlay\nTrue: {true_cls} | Pred: {pred_cls} (p={prob:.3f})", fontsize=11, weight='bold')
        axes[2].axis('off')

        # Annotate top feature pairs in plot
        top_str = "\n".join([f"• {p['feature_i']} × {p['feature_j']} (act={p['activation']:.2f})" for p in top_pairs[:3]])
        fig.text(0.5, -0.05, f"Top Contributing Feature Interactions:\n{top_str}", ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.8))

        save_filename = f"gradcam_sample_{orig_idx}_true{true_cls}_pred{pred_cls}.png"
        save_path = os.path.join(output_dir, save_filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path} (True: {true_cls}, Pred: {pred_cls}, Prob: {prob:.4f})")

        explanations_summary.append({
            "sample_idx": orig_idx,
            "category": category,
            "true_class": true_cls,
            "pred_class": pred_cls,
            "pred_prob": float(prob),
            "top_interactions": top_pairs[:3]
        })

    # Save summary gallery
    create_gallery(selected_samples, model, feature_mapping, output_dir)

    # Save explanations log
    with open(os.path.join(output_dir, "gradcam_explanations.json"), "w") as f:
        json.dump(explanations_summary, f, indent=2)
    print(f"Grad-CAM summary and explanations saved to: {output_dir}/gradcam_explanations.json")

    return explanations_summary


def create_gallery(selected_samples, model, feature_mapping, output_dir):
    """Creates a consolidated gallery of Grad-CAM overlays."""
    n_samples = len(selected_samples)
    fig, axes = plt.subplots(2, n_samples // 2 + n_samples % 2, figsize=(4 * (n_samples // 2 + 1), 8))
    axes = axes.flatten()

    for i, sample in enumerate(selected_samples):
        img_pil = Image.open(sample['file_path']).convert('RGB')
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_arr, axis=0)

        heatmap, prob, _ = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name="conv3")
        superimposed, _ = overlay_heatmap_on_image(img_arr, heatmap, alpha=0.5)

        axes[i].imshow(superimposed)
        title_color = 'green' if sample['is_correct'] else 'red'
        status = "Correct" if sample['is_correct'] else "Misclassified"
        axes[i].set_title(
            f"Sample #{sample['sample_idx']} [{status}]\nTrue:{sample['class']} | Pred:{sample['pred_class']} (p={sample['pred_prob']:.2f})",
            fontsize=9, color=title_color, weight='bold'
        )
        axes[i].axis('off')

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Grad-CAM Interpretability Gallery: ASD Cross-Modal CNN", fontsize=14, weight='bold')
    plt.tight_layout()
    gallery_path = os.path.join(output_dir, "gradcam_summary_gallery.png")
    plt.savefig(gallery_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM gallery saved to: {gallery_path}")


if __name__ == "__main__":
    run_gradcam_analysis()
