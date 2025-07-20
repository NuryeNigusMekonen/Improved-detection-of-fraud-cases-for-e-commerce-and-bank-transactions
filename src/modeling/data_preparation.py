import pandas as pd
from sklearn.model_selection import train_test_split

def load_parquet_data(file_path, target_col='class'):
    df = pd.read_parquet(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

