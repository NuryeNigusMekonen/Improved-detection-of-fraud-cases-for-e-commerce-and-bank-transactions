# src/transformers.py

import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(level=logging.INFO)

def get_preprocessor(numeric_cols, categorical_cols):
    transformers = []
    
    if numeric_cols:
        transformers.append(
            ("num", StandardScaler(), numeric_cols)
        )
        
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        )
        
    if not transformers:
        raise ValueError("No columns provided for preprocessing.")
        
    return ColumnTransformer(transformers)


def apply_balancing(X, y, strategy="smote", sampling_strategy="auto"):
    logging.info(f"Original class distribution: {y.value_counts().to_dict()}")

    if strategy == "smote":
        balancer = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    elif strategy == "undersample":
        balancer = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
    else:
        raise ValueError("Strategy must be 'smote' or 'undersample'")
    
    X_res, y_res = balancer.fit_resample(X, y)
    logging.info(f"Balanced class distribution: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def get_full_pipeline(numeric_cols, categorical_cols, balance_strategy="smote", sampling_strategy="auto"):
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    
    if balance_strategy:
        if balance_strategy not in ["smote", "undersample"]:
            raise ValueError("Invalid balancing strategy.")
        if balance_strategy == "smote":
            balancer = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
        else:
            balancer = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
            
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("balancer", balancer)
        ])
    else:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])
    
    return pipeline
