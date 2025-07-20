import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(level=logging.INFO)

def get_preprocessor(numeric_cols, categorical_cols, encoder_type='onehot'):
    transformers = []
    
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    
    if categorical_cols:
        if encoder_type == 'onehot':
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            )
        else:
            raise ValueError("Only 'onehot' encoding is supported as per task requirement.")
    
    if not transformers:
        raise ValueError("No columns provided for preprocessing.")
    
    return ColumnTransformer(transformers)

def fit_transform_to_df(preprocessor, X: pd.DataFrame):
    """
    Fit the preprocessor and return transformed DataFrame with feature names.
    """
    X_trans = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
    return X_trans_df

def transform_to_df(preprocessor, X: pd.DataFrame):
    """
    Transform with fitted preprocessor and return DataFrame with feature names.
    """
    X_trans = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
    return X_trans_df

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
