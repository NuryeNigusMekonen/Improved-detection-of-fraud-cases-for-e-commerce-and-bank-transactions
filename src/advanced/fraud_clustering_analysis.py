# src/advanced/fraud_clustering_analysis.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)

def perform_clustering_analysis(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    features = ['purchase_value', 'age', 'time_to_purchase', 'high_value_transaction', 'device_transaction_count']
    features = [f for f in features if f in df.columns]

    if not features:
        logging.warning("No valid features found for clustering.")
        df['cluster'] = -1
        return df

    try:
        X = df[features].fillna(0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        logging.info(f"Added 'cluster' column with {n_clusters} clusters.")
    except Exception as e:
        logging.warning(f"Clustering failed: {e}")
        df['cluster'] = -1

    return df
