#!/usr/bin/env python3
"""
Step 1 & Step 2: Data Acquisition and Preprocessing
Dataset: UCI Autism Screening Adult (id=426)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def acquire_data(raw_data_path="data/raw/autism_adult_raw.csv"):
    """
    Downloads UCI Autism Screening Adult dataset using ucimlrepo or fallback.
    Saves raw merged data to raw_data_path.
    """
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    print("=" * 60)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 60)
    
    df_raw = None
    try:
        from ucimlrepo import fetch_ucirepo
        print("Attempting to fetch dataset id=426 via ucimlrepo...")
        dataset = fetch_ucirepo(id=426)
        X = dataset.data.features
        y = dataset.data.targets
        # Standardize target column name
        if isinstance(y, pd.DataFrame):
            target_col = y.columns[0]
            y_series = y[target_col].rename("Class/ASD")
        else:
            y_series = pd.Series(y, name="Class/ASD")
            
        df_raw = pd.concat([X, y_series], axis=1)
        print("Successfully fetched dataset via ucimlrepo.")
    except Exception as e:
        print(f"ucimlrepo fetch failed: {e}. Attempting fallback download...")
        import urllib.request
        import zipfile
        import io
        import arff

        arff_url = "https://archive.ics.uci.edu/static/public/426/autism+screening+adult.zip"
        req = urllib.request.Request(arff_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                arff_files = [f for f in z.namelist() if f.endswith(".arff")]
                if not arff_files:
                    raise FileNotFoundError("No .arff file found in downloaded zip.")
                with z.open(arff_files[0]) as arff_file:
                    arff_data = arff.load(io.StringIO(arff_file.read().decode('utf-8')))
                    df_raw = pd.DataFrame(arff_data['data'], columns=[a[0] for a in arff_data['attributes']])
        print("Successfully downloaded and parsed ARFF data via fallback.")

    # Standardize column naming if necessary
    rename_dict = {}
    for col in df_raw.columns:
        col_lower = col.lower().strip()
        if col_lower in ['class', 'class/asd', 'class/asd']:
            rename_dict[col] = 'Class/ASD'
        elif col_lower in ['austim', 'family_pdd', 'family_history']:
            rename_dict[col] = 'family_history'
        elif col_lower == 'jundice':
            rename_dict[col] = 'jaundice'
        elif col_lower == 'gender':
            rename_dict[col] = 'gender'
        elif col_lower == 'age':
            rename_dict[col] = 'age'
    df_raw = df_raw.rename(columns=rename_dict)

    # Save raw data
    df_raw.to_csv(raw_data_path, index=False)
    print(f"Raw merged data saved to: {raw_data_path}")

    # Print data profile
    print("\n--- RAW DATA PROFILE ---")
    print(f"Shape: {df_raw.shape}")
    print("\nData Types:")
    print(df_raw.dtypes)
    print("\nMissing Values Count (raw):")
    print(df_raw.isnull().sum())
    
    return df_raw


def preprocess_data(raw_data_path="data/raw/autism_adult_raw.csv",
                    processed_data_path="data/processed/cleaned_data.csv"):
    """
    Cleans raw data, imputes missing values, encodes categorical features,
    normalizes age, saves cleaned data, and prints class balance.
    """
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)

    if not os.path.exists(raw_data_path):
        df = acquire_data(raw_data_path)
    else:
        df = pd.read_csv(raw_data_path)

    initial_rows = len(df)

    # 1. Replace '?' with NaN and clean string whitespace / quotes
    df = df.replace('?', np.nan)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip('"')
        df[col] = df[col].replace({'nan': np.nan, '?': np.nan, 'None': np.nan})

    # Convert numeric columns that might be stored as object/str
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    for i in range(1, 11):
        score_col = f"A{i}_Score"
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')

    # 2. Check missing values per row: drop only if >50% of fields are missing
    threshold = 0.5 * df.shape[1]
    missing_per_row = df.isnull().sum(axis=1)
    drop_mask = missing_per_row > threshold
    if drop_mask.sum() > 0:
        print(f"Dropping {drop_mask.sum()} rows with >50% missing fields.")
        df = df[~drop_mask].copy()
    else:
        print(f"No rows had >50% missing fields. Retained all {initial_rows} rows.")

    # 3. Handle missing values:
    # Impute numeric (age) with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                med = df[col].median()
                df[col] = df[col].fillna(med)
                print(f"Imputed numeric '{col}' nulls with median ({med:.2f})")
            else:
                mode_val = df[col].dropna().mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Imputed categorical '{col}' nulls with mode ('{mode_val}')")

    # Fix any anomalous age outliers (e.g. data entry error like age 383 -> median)
    if (df['age'] > 120).any():
        print(f"Found {int((df['age'] > 120).sum())} anomalous age > 120, replacing with median.")
        median_age = df.loc[df['age'] <= 120, 'age'].median()
        df.loc[df['age'] > 120, 'age'] = median_age

    # 4. Feature Encoding:
    # gender: M/F -> 1/0
    gender_map = {'m': 1, 'f': 0, '1': 1, '0': 0, 'male': 1, 'female': 0}
    df['gender'] = df['gender'].astype(str).str.lower().map(gender_map).fillna(0).astype(int)

    # jaundice: yes/no -> 1/0
    yn_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, 'y': 1, 'n': 0}
    df['jaundice'] = df['jaundice'].astype(str).str.lower().map(yn_map).fillna(0).astype(int)

    # family_history: yes/no -> 1/0
    df['family_history'] = df['family_history'].astype(str).str.lower().map(yn_map).fillna(0).astype(int)

    # used_app_before: yes/no -> 1/0 if present
    if 'used_app_before' in df.columns:
        df['used_app_before'] = df['used_app_before'].astype(str).str.lower().map(yn_map).fillna(0).astype(int)

    # Class/ASD: YES/NO -> 1/0
    target_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, 'y': 1, 'n': 0}
    df['Class/ASD'] = df['Class/ASD'].astype(str).str.lower().map(target_map).fillna(0).astype(int)

    # 5. Keep A1_Score through A10_Score as-is (ensure int 0/1)
    for i in range(1, 11):
        df[f'A{i}_Score'] = df[f'A{i}_Score'].astype(int)

    # 6. Normalize Age to [0,1] via min-max scaling
    scaler = MinMaxScaler()
    df['age_normalized'] = scaler.fit_transform(df[['age']])
    print(f"Age range: [{df['age'].min():.1f}, {df['age'].max():.1f}] -> Normalized [0, 1]")

    # Select the core feature set for modeling and encoding
    # 10 behavioral scores + demographics (age_normalized, gender, jaundice, family_history)
    feature_cols = [f'A{i}_Score' for i in range(1, 11)] + [
        'age_normalized', 'gender', 'jaundice', 'family_history'
    ]
    target_col = 'Class/ASD'

    cleaned_df = df[feature_cols + [target_col]].copy()

    # Save cleaned output
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    cleaned_df.to_csv(processed_data_path, index=False)
    print(f"\nCleaned processed data saved to: {processed_data_path}")
    print(f"Processed features: {feature_cols}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")

    # 7. Print class balance
    print("\n--- CLASS BALANCE ---")
    counts = cleaned_df[target_col].value_counts()
    percentages = cleaned_df[target_col].value_counts(normalize=True) * 100
    for cls_val in counts.index:
        label_name = "High Risk (YES / 1)" if cls_val == 1 else "Low Risk (NO / 0)"
        print(f"Class {cls_val} ({label_name}): {counts[cls_val]} samples ({percentages[cls_val]:.2f}%)")

    # Check imbalance beyond 60/40
    majority_pct = percentages.max()
    minority_pct = percentages.min()
    is_imbalanced = majority_pct > 60.0
    print(f"\nMajority class percentage: {majority_pct:.2f}% | Minority class: {minority_pct:.2f}%")
    if is_imbalanced:
        print("(!) FLAG: Dataset is imbalanced beyond 60/40 ratio!")
        print("    Class weighting will be applied during Step 4 (CNN training).")
    else:
        print("Dataset class balance is within 60/40 ratio.")

    return cleaned_df, is_imbalanced, feature_cols


if __name__ == "__main__":
    df_raw = acquire_data()
    cleaned_df, is_imbalanced, feature_cols = preprocess_data()
