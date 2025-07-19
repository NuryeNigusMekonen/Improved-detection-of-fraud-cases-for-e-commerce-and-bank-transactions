from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(df, contamination=0.01):
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df)
    df['outlier'] = (preds == -1).astype(int)
    return df
