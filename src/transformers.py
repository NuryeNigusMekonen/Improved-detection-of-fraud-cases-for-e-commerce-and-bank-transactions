# src/transformers.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging

logging.basicConfig(level=logging.INFO)

def get_preprocessor(numeric_cols, categorical_cols):
    """
    Returns a ColumnTransformer that scales numeric features and encodes categorical features.
    """
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor


def apply_balancing(X, y, strategy="smote"):
    """
    Applies class balancing to training data only.
    strategy: 'smote' or 'undersample'
    """
    logging.info(f"Original class distribution: {y.value_counts().to_dict()}")

    if strategy == "smote":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    elif strategy == "undersample":
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
    else:
        raise ValueError("Strategy must be either 'smote' or 'undersample'")

    logging.info(f"Balanced class distribution: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res

