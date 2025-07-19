from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np  
def calculate_transaction_risk(df):
    features = ["purchase_value", "time_to_purchase", "high_value_transaction", "device_transaction_count"]
    df = df.copy()
    df["transaction_risk"] = df[features].apply(lambda row: sum(row), axis=1)
    return df

def assign_risk_score(df):
    df = df.copy()
    bins = [-float('inf'), 50, 150, 300, float('inf')]
    labels = ["Low", "Medium", "High", "Critical"]
    df["risk_score_label"] = pd.cut(df["transaction_risk"], bins=bins, labels=labels)
    return df