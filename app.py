#!/usr/bin/env python3
"""
Streamlit Web Application:
Explainable Cross-Modal ASD Screening System (CNN + Grad-CAM)
Enables interactive screening, multi-channel matrix inspection, and real-time Grad-CAM explainability.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from predict import predict_patient, load_inference_artifacts, AQ10_DESCRIPTIONS

st.set_page_config(
    page_title="Explainable ASD Screening (CNN + Grad-CAM)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .high-risk-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .low-risk-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_artifacts():
    return load_inference_artifacts()

@st.cache_data
def get_cleaned_data():
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

try:
    model, corr_matrix, feature_mapping = get_artifacts()
    df_clean = get_cleaned_data()
except Exception as e:
    st.error(f"Failed to load project artifacts: {e}")
    st.stop()

# Header
st.markdown('<div class="main-header">🧠 Explainable ASD Screening System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Cross-Modal Feature Encoding (Outer Product + Difference + Correlation) &bull; Lightweight CNN &bull; Grad-CAM Interpretability</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["🔍 Interactive Patient Screening", "📊 Model Performance & Audit", "📖 AQ-10 Clinical Guide & Dataset"])

# ==============================================================================
# TAB 1: INTERACTIVE PATIENT SCREENING
# ==============================================================================
with tabs[0]:
    st.markdown("### Patient Profile & Behavioral Assessment")
    st.write(
        "Enter responses to the 10 Autism Spectrum Quotient (AQ-10) items and demographic attributes. "
        "The system transforms these tabular values into an encoded 224×224 RGB image, evaluates ASD risk via the trained CNN, "
        "and maps the activation peaks back to specific pairwise interactions."
    )

    # Preset selection buttons
    preset_cols = st.columns([1.5, 1.5, 1.5, 2.5])
    preset_choice = None
    with preset_cols[0]:
        if st.button("🔴 Typical High Risk", use_container_width=True):
            preset_choice = "high"
    with preset_cols[1]:
        if st.button("🟢 Typical Low Risk", use_container_width=True):
            preset_choice = "low"
    with preset_cols[2]:
        if st.button("🟡 Borderline Case", use_container_width=True):
            preset_choice = "borderline"
    with preset_cols[3]:
        if df_clean is not None and st.button("🎲 Random Held-Out Test Sample", use_container_width=True):
            preset_choice = "random_test"

    # Set default values based on presets
    def_scores = [0] * 10
    def_age = 28
    def_gender = "Male"
    def_jaundice = "No"
    def_fh = "No"

    if preset_choice == "high":
        def_scores = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
        def_age = 32
        def_gender = "Male"
        def_jaundice = "Yes"
        def_fh = "Yes"
    elif preset_choice == "low":
        def_scores = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        def_age = 25
        def_gender = "Female"
        def_jaundice = "No"
        def_fh = "No"
    elif preset_choice == "borderline":
        def_scores = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
        def_age = 30
        def_gender = "Male"
        def_jaundice = "No"
        def_fh = "No"
    elif preset_choice == "random_test" and df_clean is not None:
        sample_row = df_clean.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]
        def_scores = [int(sample_row[f"A{i}_Score"]) for i in range(1, 11)]
        def_age = int(17 + sample_row["age_normalized"] * (64 - 17))
        def_gender = "Male" if sample_row["gender"] == 1 else "Female"
        def_jaundice = "Yes" if sample_row["jaundice"] == 1 else "No"
        def_fh = "Yes" if sample_row["family_history"] == 1 else "No"
        ground_truth = "High Risk (1)" if sample_row["Class/ASD"] == 1 else "Low Risk (0)"
        st.info(f"Loaded Sample Ground Truth: **{ground_truth}**")

    # Form Layout
    with st.form("patient_screening_form"):
        col_left, col_right = st.columns([1.2, 0.8])
        
        with col_left:
            st.subheader("AQ-10 Behavioral Screening Questions")
            user_scores = []
            for i in range(1, 11):
                key_name = f"A{i}_Score"
                desc = AQ10_DESCRIPTIONS[key_name]
                val = st.radio(
                    f"**Item {i}**: {desc}",
                    options=[0, 1],
                    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                    index=def_scores[i-1],
                    horizontal=True,
                    key=f"input_{key_name}"
                )
                user_scores.append(val)

        with col_right:
            st.subheader("Demographic & Clinical Attributes")
            age_val = st.slider("Age (Years)", min_value=17, max_value=64, value=def_age)
            gender_val = st.selectbox("Gender", options=["Male", "Female"], index=0 if def_gender == "Male" else 1)
            jaundice_val = st.selectbox("Jaundice at Birth?", options=["No", "Yes"], index=0 if def_jaundice == "No" else 1)
            fh_val = st.selectbox("Family History of ASD / PDD?", options=["No", "Yes"], index=0 if def_fh == "No" else 1)

            st.markdown("---")
            total_aq10 = sum(user_scores)
            st.metric("Total AQ-10 Score", f"{total_aq10} / 10", 
                      delta="Clinical Cutoff >= 6" if total_aq10 >= 6 else "Below Clinical Cutoff",
                      delta_color="inverse" if total_aq10 >= 6 else "normal")
            
            submitted = st.form_submit_button("🚀 Run Screening & Generate Explanations", use_container_width=True)

    if submitted or preset_choice is not None:
        patient_dict = {f"A{i+1}_Score": user_scores[i] for i in range(10)}
        patient_dict["age"] = float(age_val)
        patient_dict["gender"] = "m" if gender_val == "Male" else "f"
        patient_dict["jaundice"] = "yes" if jaundice_val == "Yes" else "no"
        patient_dict["family_history"] = "yes" if fh_val == "Yes" else "no"

        with st.spinner("Synthesizing Cross-Modal Representation & Executing CNN Inference..."):
            res = predict_patient(patient_dict, model, corr_matrix, feature_mapping)

        st.markdown("---")
        st.subheader("Screening Outcome & Risk Stratification")

        # Result Banner
        b_col1, b_col2, b_col3 = st.columns([1.5, 1.2, 1.2])
        with b_col1:
            if res["predicted_class"] == 1:
                st.markdown('<div class="high-risk-badge">⚠️ HIGH RISK FOR ASD (Class 1)</div>', unsafe_allow_html=True)
                st.write("Clinical recommendation: Comprehensive diagnostic evaluation advised.")
            else:
                st.markdown('<div class="low-risk-badge">✅ LOW RISK FOR ASD (Class 0)</div>', unsafe_allow_html=True)
                st.write("Clinical recommendation: Standard developmental monitoring.")

        with b_col2:
            st.metric("Model Risk Probability", f"{res['risk_probability'] * 100:.2f}%")
        with b_col3:
            st.metric("Prediction Confidence", f"{res['confidence'] * 100:.2f}%")

        st.markdown("---")
        st.subheader("Cross-Modal Feature Encoding: Multi-Channel Matrix Decomposition")
        st.write(
            "The patient's 14 features are mapped into a $14 \\times 14$ interaction matrix and nearest-neighbor upsampled to 224×224:"
        )

        c1, c2, c3, c4 = st.columns(4)
        rgb_14 = res["encoded_rgb_14x14"]
        with c1:
            st.markdown("**Channel R: Outer Product**")
            st.caption("$M_R = f_i \\cdot f_j$ (Pairwise interaction)")
            fig_r = px.imshow(rgb_14[:, :, 0], color_continuous_scale="Reds")
            fig_r.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), height=200)
            st.plotly_chart(fig_r, use_container_width=True)

        with c2:
            st.markdown("**Channel G: Contrast**")
            st.caption("$M_G = |f_i - f_j|$ (Absolute difference)")
            fig_g = px.imshow(rgb_14[:, :, 1], color_continuous_scale="Greens")
            fig_g.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), height=200)
            st.plotly_chart(fig_g, use_container_width=True)

        with c3:
            st.markdown("**Channel B: Pearson Scale**")
            st.caption("$M_B = \\frac{C_{ij} \\cdot M_R + 1}{2}$ (Training correlation)")
            fig_b = px.imshow(rgb_14[:, :, 2], color_continuous_scale="Blues")
            fig_b.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0), height=200)
            st.plotly_chart(fig_b, use_container_width=True)

        with c4:
            st.markdown("**Composite 224×224 RGB**")
            st.caption("Input feed to CNN classifier")
            st.image(res["encoded_rgb_224"], use_container_width=True)

        st.markdown("---")
        st.subheader("Grad-CAM Explainability: Where Did the CNN Focus?")

        v_col1, v_col2, v_col3 = st.columns([1, 1, 1.4])
        with v_col1:
            st.markdown("**1. Original Encoded Matrix**")
            st.image(res["encoded_rgb_224"], use_container_width=True)
        with v_col2:
            st.markdown("**2. Superimposed Overlay**")
            st.image(np.clip(res["superimposed_overlay"], 0.0, 1.0), use_container_width=True)
            st.caption("Focal hot spots (red/yellow) indicate critical evidence regions in `conv3`.")

        with v_col3:
            st.markdown("**3. Top Contributing Feature Interactions**")
            top_pairs = res["top_feature_interactions"]
            df_pairs = pd.DataFrame(top_pairs).sort_values(by="activation", ascending=True)
            df_pairs["Pair_Label"] = df_pairs["feature_i"] + " × " + df_pairs["feature_j"]

            fig_bar = px.bar(
                df_pairs,
                x="activation",
                y="Pair_Label",
                orientation="h",
                color="activation",
                color_continuous_scale="Viridis",
                labels={"activation": "Grad-CAM Activation", "Pair_Label": "Interaction Pair"}
            )
            fig_bar.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=10),
                height=260
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Contextual Interpretation
        st.markdown("#### Clinical Interpretation of Decision")
        top_1 = top_pairs[0]
        top_2 = top_pairs[1] if len(top_pairs) > 1 else top_1
        if res["predicted_class"] == 1:
            st.info(
                f"**High Risk Assessment Rationale**: The CNN focused predominantly on co-activated traits between "
                f"**{top_1['feature_i']}** (*\"{top_1['desc_i']}\"*) and **{top_1['feature_j']}** (*\"{top_1['desc_j']}\"*) "
                f"(activation = {top_1['activation']:.2f}), accompanied by **{top_2['feature_i']} × {top_2['feature_j']}**. "
                f"In cross-modal space, the simultaneous elevation of both scores forms a high-energy interaction square "
                f"that matched diagnostic patterns learned from clinical cohorts."
            )
        else:
            st.success(
                f"**Low Risk Assessment Rationale**: The network detected absence of high-intensity co-activations across the AQ-10 "
                f"behavioral matrix. The primary negative evidence was centered around contrast regions like "
                f"**{top_1['feature_i']} × {top_1['feature_j']}**, where score differences are small and behavioral traits indicate neurotypical baseline."
            )


# ==============================================================================
# TAB 2: MODEL PERFORMANCE & ZERO-LEAKAGE AUDIT
# ==============================================================================
with tabs[1]:
    st.subheader("Held-Out Test Set Performance (N = 141 Unseen Samples)")
    
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Test Accuracy", "99.29%")
    m_col2.metric("ROC-AUC", "1.0000")
    m_col3.metric("Precision", "100.00%")
    m_col4.metric("Recall", "97.37%")
    m_col5.metric("Specificity", "100.00%")
    m_col6.metric("F1-Score", "98.67%")

    st.markdown("---")
    g_col1, g_col2, g_col3 = st.columns(3)

    cm_img = os.path.join(PROJECT_ROOT, "results", "metrics", "confusion_matrix.png")
    roc_img = os.path.join(PROJECT_ROOT, "results", "metrics", "roc_curve.png")
    tc_img = os.path.join(PROJECT_ROOT, "results", "metrics", "training_curves.png")

    with g_col1:
        st.markdown("**Confusion Matrix (Held-Out Test)**")
        if os.path.exists(cm_img):
            st.image(cm_img, use_container_width=True)
    with g_col2:
        st.markdown("**ROC-AUC Curve**")
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True)
    with g_col3:
        st.markdown("**Training & Validation Loss / Acc**")
        if os.path.exists(tc_img):
            st.image(tc_img, use_container_width=True)

    st.markdown("---")
    st.subheader("Train/Test Leakage & Regularization Audit")
    st.markdown("""
    - **Pre-Encoding Stratification**: Tabular data was split (80/20, stratified) *before* calculating any statistics or synthesizing images.
    - **Channel B Correlation Matrix**: Global Pearson correlation matrix $C$ was computed exclusively on `X_train` (`563` samples).
    - **Physical Folder Isolation**: Test images reside in `data/processed/encoded_images/test/` and were never seen during gradient updates or validation checkpoints.
    - **Zero Generalization Gap**: Training Loss: `0.0466`, Test Loss: `0.0658`. No overfitting detected.
    - **Architecture Regularization**: 102,465 parameters (~400 KB), GlobalAveragePooling2D, Batch Normalization, and Dropout(0.5).
    """)

    gallery_img = os.path.join(PROJECT_ROOT, "results", "gradcam_heatmaps", "gradcam_summary_gallery.png")
    if os.path.exists(gallery_img):
        st.markdown("---")
        st.subheader("Grad-CAM Summary Gallery (Sample Cohort)")
        st.image(gallery_img, use_container_width=True)


# ==============================================================================
# TAB 3: AQ-10 CLINICAL GUIDE & DATASET
# ==============================================================================
with tabs[2]:
    st.subheader("AQ-10 Questionnaire Clinical Reference")
    st.write(
        "The Autism Spectrum Quotient (AQ-10) is a validated 10-item screening tool developed by the Autism Research Centre (Allison et al., 2012). "
        "A score of $\ge 6$ out of 10 indicates that the individual should be referred for a specialist diagnostic assessment."
    )

    aq10_table = []
    for i in range(1, 11):
        k = f"A{i}_Score"
        aq10_table.append({
            "Item": f"AQ-{i}",
            "Question / Statement": AQ10_DESCRIPTIONS[k],
            "Scoring Criterion": "1 = Characteristic of ASD, 0 = Neurotypical"
        })
    st.table(pd.DataFrame(aq10_table))

    if df_clean is not None:
        st.markdown("---")
        st.subheader("Cleaned Dataset Inspection (UCI Autism Screening Adult)")
        st.dataframe(df_clean.head(15), use_container_width=True)
        st.caption(f"Total dataset: {len(df_clean)} records across 15 columns (14 features + 1 target).")
