from sklearn.ensemble import RandomForestClassifier

def compute_fraud_risk_score(X_train, y_train, X_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    risk_scores = model.predict_proba(X_test)[:, 1]  # probability of fraud
    return risk_scores
