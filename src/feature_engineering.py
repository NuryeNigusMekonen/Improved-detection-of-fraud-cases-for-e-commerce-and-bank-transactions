# src/feature_engineering.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "signup_time" in df.columns and "purchase_time" in df.columns:
        df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600
        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.dayofweek
        logging.info("Added time_since_signup, hour_of_day, and day_of_week features.")
    else:
        logging.warning("Required datetime columns missing for time feature engineering.")
    return df


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    if "user_id" in df.columns:
        df["user_transaction_count"] = df.groupby("user_id")["user_id"].transform("count")
        logging.info("Added user_transaction_count feature.")
    if "device_id" in df.columns:
        df["device_transaction_count"] = df.groupby("device_id")["device_id"].transform("count")
        logging.info("Added device_transaction_count feature.")
    return df


import logging

def merge_ip_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Convert fraud_df ip_address floats to integer safely
        fraud_df = fraud_df.copy()
        fraud_df["ip_int"] = fraud_df["ip_address"].apply(lambda x: int(x) if pd.notnull(x) else None)
        fraud_df = fraud_df.dropna(subset=["ip_int"])

        # Ensure ip_df bounds are integers
        ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype(int)
        ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype(int)
        ip_df = ip_df.sort_values(by=["lower_bound_ip_address", "upper_bound_ip_address"])

        def find_country(ip_val):
            match = ip_df.loc[
                (ip_df["lower_bound_ip_address"] <= ip_val) &
                (ip_val <= ip_df["upper_bound_ip_address"]),
                "country"
            ]
            return match.iloc[0] if not match.empty else "Unknown"

        fraud_df["country"] = fraud_df["ip_int"].apply(find_country)

    except Exception as e:
        logging.warning(f"IP merge failed: {e}")
        fraud_df["country"] = "Unknown"

    # Drop the temporary ip_int column
    return fraud_df.drop(columns=["ip_int"])



def save_feature_engineered_data(df: pd.DataFrame, filename="fraud_data_features.parquet", output_dir="../data/processed/"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved feature-engineered data to {output_path}")
