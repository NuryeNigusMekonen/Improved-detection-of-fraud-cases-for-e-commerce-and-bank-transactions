# src/preprocessing.py

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)


def report_missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        logging.info(f"Missing values summary:\n{missing}")
    else:
        logging.info("No missing values detected.")


def handle_missing_values(df: pd.DataFrame, drop_threshold=0.5, fillna_numeric=True) -> pd.DataFrame:
    logging.info(f"Initial DataFrame shape: {df.shape}")
    report_missing_values(df)

    # Drop columns with missing percentage > threshold
    threshold = drop_threshold * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    logging.info(f"After dropping columns with >{drop_threshold*100}% missing values: {df.shape}")

    # Fill numeric columns with median
    if fillna_numeric:
        for col in df.select_dtypes(include=[np.number]).columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Fill categorical columns with mode or 'Unknown'
    for col in df.select_dtypes(include=['object', 'category']).columns:
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode)

    report_missing_values(df)
    return df


def clean_data_types(df: pd.DataFrame, datetime_cols: list = None) -> pd.DataFrame:
    if datetime_cols:
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logging.info(f"Converted column '{col}' to datetime.")
            except Exception as e:
                logging.warning(f"Failed to convert '{col}': {e}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    logging.info(f"Removed {before - after} duplicate rows.")
    return df


def save_cleaned_data(df: pd.DataFrame, filename="fraud_data_cleaned.parquet", output_dir="../data/processed/"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved cleaned data to {output_path}")
