import pandas as pd
import numpy as np

def add_time_deltas(df):
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600  # hours
    return df
def add_transaction_amount_flags(df, threshold=100):
    df['high_value_transaction'] = (df['purchase_value'] > threshold).astype(int)
    return df

def add_device_transaction_counts(df):
    device_counts = df['device_id'].value_counts()
    df['device_transaction_count'] = df['device_id'].map(device_counts)
    return df