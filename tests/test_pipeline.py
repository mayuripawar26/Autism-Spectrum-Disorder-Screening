#!/usr/bin/env python3
"""
Comprehensive automated unit and regression tests for the ASD Cross-Modal Screening pipeline.
Tests data preprocessing, matrix encoding, model architecture, inference, Grad-CAM, and zero leakage.
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cross_modal_encoding import encode_feature_vector
from model import build_custom_cnn
from predict import predict_patient, load_inference_artifacts


class TestASDPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cleaned_csv = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")
        cls.train_csv = os.path.join(PROJECT_ROOT, "data", "processed", "train_data.csv")
        cls.test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "test_data.csv")
        cls.corr_npy = os.path.join(PROJECT_ROOT, "data", "processed", "train_correlation_matrix.npy")
        cls.model_path = os.path.join(PROJECT_ROOT, "models", "cnn_asd_model.keras")

    def test_01_cleaned_data_integrity(self):
        """Verifies cleaned dataset exists, has no NaNs, and correct schema."""
        self.assertTrue(os.path.exists(self.cleaned_csv), "cleaned_data.csv does not exist.")
        df = pd.read_csv(self.cleaned_csv)
        self.assertEqual(len(df), 704, "Cleaned dataset should have 704 rows.")
        self.assertEqual(df.isnull().sum().sum(), 0, "Cleaned dataset contains NaN values.")
        self.assertIn("Class/ASD", df.columns, "Target column 'Class/ASD' missing.")
        self.assertTrue((df["age_normalized"] >= 0.0).all() and (df["age_normalized"] <= 1.0).all(),
                        "Age normalized values out of [0, 1] range.")

    def test_02_zero_data_leakage(self):
        """Ensures training and test set indices are strictly disjoint."""
        self.assertTrue(os.path.exists(self.train_csv), "train_data.csv missing.")
        self.assertTrue(os.path.exists(self.test_csv), "test_data.csv missing.")
        train_df = pd.read_csv(self.train_csv, index_col=0)
        test_df = pd.read_csv(self.test_csv, index_col=0)
        overlap = set(train_df.index).intersection(set(test_df.index))
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Overlapping indices: {overlap}")
        self.assertEqual(len(train_df), 563, "Train split should have 563 samples.")
        self.assertEqual(len(test_df), 141, "Test split should have 141 samples.")

    def test_03_cross_modal_encoding_shape_and_bounds(self):
        """Verifies mathematical properties of cross-modal encoding."""
        self.assertTrue(os.path.exists(self.corr_npy), "Correlation matrix missing.")
        corr = np.load(self.corr_npy)
        self.assertEqual(corr.shape, (14, 14), "Correlation matrix must be 14x14.")
        
        # Test dummy feature vector of 14 elements
        dummy_f = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0.5, 1, 0, 1], dtype=np.float32)
        encoded = encode_feature_vector(dummy_f, corr)
        self.assertEqual(encoded.shape, (14, 14, 3), "Encoded matrix must be shape (14, 14, 3).")
        self.assertTrue((encoded >= 0.0).all() and (encoded <= 1.0).all(),
                        "Encoded RGB matrix contains values outside [0, 1].")

    def test_04_cnn_architecture(self):
        """Verifies CNN layer structure and parameter count."""
        cnn = build_custom_cnn(input_shape=(224, 224, 3))
        layer_names = [l.name for l in cnn.layers]
        self.assertIn("conv1", layer_names)
        self.assertIn("conv2", layer_names)
        self.assertIn("conv3", layer_names)
        self.assertIn("global_avg_pool", layer_names)
        self.assertIn("output_pred", layer_names)
        self.assertLess(cnn.count_params(), 120000, "Model should be lightweight (< 120k params).")

    def test_05_end_to_end_prediction_and_gradcam(self):
        """Verifies prediction API and Grad-CAM generation on test patient."""
        model, corr, mapping = load_inference_artifacts()
        sample_patient = {
            "A1_Score": 1, "A2_Score": 1, "A3_Score": 1, "A4_Score": 1, "A5_Score": 1,
            "A6_Score": 1, "A7_Score": 0, "A8_Score": 1, "A9_Score": 1, "A10_Score": 1,
            "age": 30.0, "gender": "m", "jaundice": "yes", "family_history": "yes"
        }
        res = predict_patient(sample_patient, model, corr, mapping)
        self.assertIn(res["predicted_class"], [0, 1])
        self.assertTrue(0.0 <= res["risk_probability"] <= 1.0)
        self.assertEqual(res["encoded_rgb_224"].shape, (224, 224, 3))
        self.assertEqual(res["gradcam_heatmap"].shape, (224, 224))
        self.assertGreaterEqual(len(res["top_feature_interactions"]), 3)


if __name__ == "__main__":
    unittest.main()
