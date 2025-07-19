# src/advanced/outlier_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest
import logging

logging.basicConfig(level=logging.INFO)

def detect_outliers_iforest(df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    features = ['purchase_value', 'age', 'time_to_purchase', 'device_transaction_count']
    features = [f for f in features if f in df.columns]

    if not features:
        logging.warning("No valid features found for outlier detection.")
        df['outlier'] = 0
        return df

    try:
        X = df[features].fillna(0)
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df['outlier'] = iso_forest.fit_predict(X)
        df['outlier'] = (df['outlier'] == -1).astype(int)
        logging.info(f"Outlier detection completed. Found {df['outlier'].sum()} outliers.")
    except Exception as e:
        logging.warning(f"Outlier detection failed: {e}")
        df['outlier'] = 0

    return df
