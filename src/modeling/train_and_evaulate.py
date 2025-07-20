from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import logging

logging.basicConfig(level=logging.INFO)

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Trained Logistic Regression.")
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Trained Random Forest Classifier.")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logging.info(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "report": report, "conf_matrix": cm}

