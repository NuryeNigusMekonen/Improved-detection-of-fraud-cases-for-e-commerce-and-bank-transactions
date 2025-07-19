import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)


def add_time_deltas(df):
    df = df.copy()
    try:
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        logging.info(" Added 'time_to_purchase'")
    except Exception as e:
        logging.error(f" Error in add_time_deltas: {e}")
    return df


def add_transaction_amount_flags(df, threshold=100):
    df = df.copy()
    try:
        df['high_value_transaction'] = (df['purchase_value'] > threshold).astype(int)
        logging.info(f" Added 'high_value_transaction' (threshold={threshold})")
    except Exception as e:
        logging.error(f" Error in add_transaction_amount_flags: {e}")
    return df


def add_device_transaction_counts(df):
    df = df.copy()
    try:
        device_counts = df['device_id'].value_counts()
        df['device_transaction_count'] = df['device_id'].map(device_counts)
        logging.info(" Added 'device_transaction_count'")
    except Exception as e:
        logging.error(f" Error in add_device_transaction_counts: {e}")
    return df


def plot_feature_histograms(df, numeric_cols, save=False, folder="../reports/figures/adv_features/"):
    for col in numeric_cols:
        try:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), bins=40, kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            if save:
                plt.savefig(f"{folder}{col}_distribution.png", dpi=300)
            plt.show()
            logging.info(f" Plotted {col} distribution.")
        except Exception as e:
            logging.error(f" Error plotting {col}: {e}")


def apply_advanced_feature_engineering_with_plots(df, threshold=100, save_plots=False):
    df = add_time_deltas(df)
    df = add_transaction_amount_flags(df, threshold)
    df = add_device_transaction_counts(df)
    numeric_cols = ['purchase_value', 'age', 'time_to_purchase', 'device_transaction_count']
    plot_feature_histograms(df, numeric_cols, save=save_plots)
    logging.info(" Advanced feature engineering with plots completed.")
    return df
