import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def add_transaction_amount_flags(df: pd.DataFrame, threshold: float = 100) -> pd.DataFrame:
    try:
        df = df.copy()
        df["high_value_transaction"] = (df["purchase_value"] > threshold).astype(int)
        logging.info(f"Added high_value_transaction with threshold {threshold}.")
    except Exception as e:
        logging.warning(f"Failed to add high_value_transaction: {e}")
    return df

def add_time_to_purchase(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        df["time_to_purchase"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600  # hours
        logging.info("Added time_to_purchase feature.")
    except Exception as e:
        logging.warning(f"Failed to add time_to_purchase: {e}")
    return df

def save_advanced_featured_data(df: pd.DataFrame, filename="fraud_data_advanced_features.parquet", output_dir="../data/processed/"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved advanced feature-engineered data to {output_path}")
