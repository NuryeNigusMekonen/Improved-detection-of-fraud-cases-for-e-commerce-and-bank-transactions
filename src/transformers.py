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
            raise ValueError("Only 'onehot' encoding is supported.")
    if not transformers:
        raise ValueError("No columns provided for preprocessing.")
    return ColumnTransformer(transformers)


def fit_transform_to_df(preprocessor, X: pd.DataFrame, drop_constant=True):
    X_trans = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)

    if drop_constant:
        dropped_cols = X_trans_df.columns[(X_trans_df.nunique() <= 1) | (X_trans_df.isna().all())]
        X_trans_df = X_trans_df.drop(columns=dropped_cols)
        logging.info(f"Dropped {len(dropped_cols)} constant/NaN columns: {list(dropped_cols)}")
    return X_trans_df


def transform_to_df(preprocessor, X: pd.DataFrame, selected_features=None):
    X_trans = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)

    if selected_features is not None:
        missing = set(selected_features) - set(X_trans_df.columns)
        if missing:
            raise ValueError(f"Missing columns in transformed data: {missing}")
        X_trans_df = X_trans_df[selected_features]
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
        balancer = SMOTE(random_state=42, sampling_strategy=sampling_strategy) \
            if balance_strategy == "smote" else \
            RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("balancer", balancer)
        ])
    else:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])
    return pipeline
