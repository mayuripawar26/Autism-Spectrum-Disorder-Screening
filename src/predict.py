#!/usr/bin/env python3
"""
CLI and Python API for Real-Time Single-Patient ASD Risk Screening & Explainability.
Encodes patient behavioral/demographic attributes into 224x224 cross-modal RGB image,
runs inference with the trained CNN model, and computes Grad-CAM feature attribution.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tensorflow as tf

# Add project root and src directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from cross_modal_encoding import encode_feature_vector
from gradcam import make_gradcam_heatmap, map_heatmap_to_features, overlay_heatmap_on_image

# Clinical descriptions for AQ-10 questions
AQ10_DESCRIPTIONS = {
    "A1_Score": "I often notice small sounds when others do not.",
    "A2_Score": "I usually concentrate more on the whole picture, rather than small details.",
    "A3_Score": "I find it easy to do more than one thing at once.",
    "A4_Score": "If there is an interruption, I can switch back to what I was doing very quickly.",
    "A5_Score": "I find it easy to 'read between the lines' when someone is talking to me.",
    "A6_Score": "I know how to tell if someone listening to me is getting bored.",
    "A7_Score": "When I'm reading a story, I find it difficult to work out the characters' intentions.",
    "A8_Score": "I like to collect information about categories of things.",
    "A9_Score": "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    "A10_Score": "I find it difficult to work out people's intentions."
}

def load_inference_artifacts(
    model_path=os.path.join(PROJECT_ROOT, "models", "cnn_asd_model.keras"),
    corr_path=os.path.join(PROJECT_ROOT, "data", "processed", "train_correlation_matrix.npy"),
    mapping_path=os.path.join(PROJECT_ROOT, "data", "processed", "feature_mapping.json")
):
    """Loads trained CNN model, training correlation matrix, and feature mapping."""
    if not os.path.exists(model_path):
        alt_path = model_path.replace(".keras", ".h5")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}")

    model = tf.keras.models.load_model(model_path)
    corr_matrix = np.load(corr_path)
    with open(mapping_path, "r") as f:
        feature_mapping = json.load(f)

    return model, corr_matrix, feature_mapping


def predict_patient(
    feature_dict,
    model=None,
    corr_matrix=None,
    feature_mapping=None,
    output_dir=None
):
    """
    Performs end-to-end inference for a single patient profile.

    Parameters:
      feature_dict: dict containing:
        - A1_Score through A10_Score (int 0 or 1)
        - age (float) or age_normalized (float in [0, 1])
        - gender ('m'/'f' or 1/0)
        - jaundice ('yes'/'no' or 1/0)
        - family_history ('yes'/'no' or 1/0)
      model: pre-loaded Keras model (optional)
      corr_matrix: pre-loaded correlation matrix (optional)
      feature_mapping: pre-loaded feature mapping dict (optional)
      output_dir: directory to save visual explanations (optional)

    Returns:
      dict with prediction probabilities, binary classification,
      encoded image arrays, Grad-CAM heatmap, and top interacting feature pairs.
    """
    if model is None or corr_matrix is None or feature_mapping is None:
        model, corr_matrix, feature_mapping = load_inference_artifacts()

    # Normalize inputs
    vec = []
    for i in range(1, 11):
        key = f"A{i}_Score"
        val = int(feature_dict.get(key, 0))
        vec.append(val)

    # Age normalization (using min 17, max 64 from dataset)
    if "age_normalized" in feature_dict:
        age_norm = float(feature_dict["age_normalized"])
    else:
        raw_age = float(feature_dict.get("age", 27.0))
        age_norm = float(np.clip((raw_age - 17.0) / (64.0 - 17.0), 0.0, 1.0))
    vec.append(age_norm)

    # Gender: m=1, f=0
    g = str(feature_dict.get("gender", "m")).strip().lower()
    vec.append(1.0 if g in ["1", "m", "male"] else 0.0)

    # Jaundice: yes=1, no=0
    j = str(feature_dict.get("jaundice", "no")).strip().lower()
    vec.append(1.0 if j in ["1", "yes", "y"] else 0.0)

    # Family history: yes=1, no=0
    fh = str(feature_dict.get("family_history", "no")).strip().lower()
    vec.append(1.0 if fh in ["1", "yes", "y"] else 0.0)

    feature_array = np.array(vec, dtype=np.float32)

    # 1. Cross-Modal Feature Encoding
    rgb_14x14 = encode_feature_vector(feature_array, corr_matrix)
    img_uint8 = np.clip(rgb_14x14 * 255.0, 0, 255).astype(np.uint8)
    img_224 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_NEAREST)
    img_normalized = img_224.astype(np.float32) / 255.0

    # 2. CNN Prediction
    input_tensor = np.expand_dims(img_normalized, axis=0)
    prob = float(model.predict(input_tensor, verbose=0)[0][0])
    pred_class = 1 if prob >= 0.5 else 0
    risk_label = "High Risk (Class 1)" if pred_class == 1 else "Low Risk (Class 0)"

    # 3. Grad-CAM Explanation
    heatmap, _, explained_class = make_gradcam_heatmap(
        input_tensor, model, last_conv_layer_name="conv3", pred_index=pred_class
    )
    superimposed, color_heatmap = overlay_heatmap_on_image(img_normalized, heatmap, alpha=0.5)
    grid_heatmap, top_pairs = map_heatmap_to_features(heatmap, feature_mapping, grid_size=14)

    # Enrich top pairs with clinical context
    for pair in top_pairs:
        f_i = pair["feature_i"]
        f_j = pair["feature_j"]
        pair["desc_i"] = AQ10_DESCRIPTIONS.get(f_i, f_i)
        pair["desc_j"] = AQ10_DESCRIPTIONS.get(f_j, f_j)

    heatmap_224 = cv2.resize(heatmap, (224, 224))

    result = {
        "risk_label": risk_label,
        "predicted_class": pred_class,
        "confidence": prob if pred_class == 1 else (1.0 - prob),
        "risk_probability": prob,
        "feature_vector": feature_array.tolist(),
        "encoded_rgb_14x14": rgb_14x14,
        "encoded_rgb_224": img_224,
        "gradcam_heatmap": heatmap_224,
        "gradcam_heatmap_raw": heatmap,
        "superimposed_overlay": superimposed,
        "top_feature_interactions": top_pairs
    }

    # Save to disk if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save raw encoded image
        Image.fromarray(img_224, mode="RGB").save(os.path.join(output_dir, "encoded_image.png"))
        # Save overlay
        overlay_uint8 = np.uint8(np.clip(superimposed * 255.0, 0, 255))
        Image.fromarray(overlay_uint8, mode="RGB").save(os.path.join(output_dir, "gradcam_overlay.png"))
        # Save JSON report
        report_data = {
            "risk_label": risk_label,
            "predicted_class": pred_class,
            "risk_probability": float(prob),
            "top_feature_interactions": top_pairs
        }
        with open(os.path.join(output_dir, "prediction_report.json"), "w") as f:
            json.dump(report_data, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="ASD Screening Cross-Modal CNN Predictor & Explainability Tool")
    parser.add_argument("--sample-idx", type=int, help="Screen an existing sample from data/processed/cleaned_data.csv")
    parser.add_argument("--scores", type=str, help="Comma-separated 10 AQ-10 scores (e.g., '1,1,1,1,1,0,1,1,1,1')")
    parser.add_argument("--age", type=float, default=28.0, help="Age in years (e.g., 28.0)")
    parser.add_argument("--gender", type=str, default="m", choices=["m", "f"], help="Gender ('m' or 'f')")
    parser.add_argument("--jaundice", type=str, default="no", choices=["yes", "no"], help="Jaundice at birth ('yes' or 'no')")
    parser.add_argument("--family-history", type=str, default="no", choices=["yes", "no"], help="Family history of ASD ('yes' or 'no')")
    parser.add_argument("--output-dir", type=str, default="results/single_prediction", help="Directory to save explanation images")
    args = parser.parse_args()

    model, corr_matrix, feature_mapping = load_inference_artifacts()

    if args.sample_idx is not None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")
        df = pd.read_csv(csv_path)
        if args.sample_idx < 0 or args.sample_idx >= len(df):
            print(f"Error: Sample index {args.sample_idx} is out of bounds (0 - {len(df) - 1})")
            sys.exit(1)
        row = df.iloc[args.sample_idx]
        feature_dict = row.to_dict()
        true_label = row.get("Class/ASD", "Unknown")
        print(f"\nEvaluating Dataset Sample #{args.sample_idx} (Ground Truth: {true_label})")
    elif args.scores is not None:
        raw_scores = [int(s.strip()) for s in args.scores.split(",")]
        if len(raw_scores) != 10:
            print("Error: --scores must provide exactly 10 comma-separated integers (0 or 1).")
            sys.exit(1)
        feature_dict = {f"A{i+1}_Score": raw_scores[i] for i in range(10)}
        feature_dict["age"] = args.age
        feature_dict["gender"] = args.gender
        feature_dict["jaundice"] = args.jaundice
        feature_dict["family_history"] = args.family_history
        print("\nEvaluating Custom Patient Profile...")
    else:
        # Default demo profile (Typical High-Risk profile)
        print("\nNo inputs provided. Running demo on a representative profile...")
        feature_dict = {
            "A1_Score": 1, "A2_Score": 1, "A3_Score": 1, "A4_Score": 1, "A5_Score": 1,
            "A6_Score": 1, "A7_Score": 0, "A8_Score": 1, "A9_Score": 1, "A10_Score": 1,
            "age": 30.0, "gender": "m", "jaundice": "yes", "family_history": "yes"
        }

    res = predict_patient(feature_dict, model, corr_matrix, feature_mapping, output_dir=args.output_dir)

    print("=" * 60)
    print("ASD SCREENING RESULT & EXPLANATION")
    print("=" * 60)
    print(f"Predicted Diagnosis:  {res['risk_label']}")
    print(f"Risk Probability:     {res['risk_probability'] * 100:.2f}%")
    print(f"Decision Confidence:  {res['confidence'] * 100:.2f}%")
    print("\nTop Contributing Behavioral & Demographic Interactions (Grad-CAM):")
    for i, pair in enumerate(res['top_feature_interactions'][:3], 1):
        print(f"  {i}. {pair['feature_i']} × {pair['feature_j']} (Activation: {pair['activation']:.3f})")
        print(f"     -> {pair['desc_i']}")
        print(f"     -> {pair['desc_j']}")
    print(f"\nVisual explanations saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
