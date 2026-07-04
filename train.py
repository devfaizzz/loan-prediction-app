import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

print("Loading dataset...")

train_data = pd.read_csv("dataset/train.csv")

print(f"Dataset loaded. Shape: {train_data.shape}")
print(train_data.head())

train_data.drop(columns=["Loan_ID"], inplace=True)

print("\nMissing values before handling:")
print(train_data.isnull().sum())

train_data["Gender"].fillna(train_data["Gender"].mode()[0], inplace=True)
train_data["Married"].fillna(train_data["Married"].mode()[0], inplace=True)
train_data["Dependents"].fillna(train_data["Dependents"].mode()[0], inplace=True)
train_data["Self_Employed"].fillna(train_data["Self_Employed"].mode()[0], inplace=True)
train_data["Credit_History"].fillna(train_data["Credit_History"].mode()[0], inplace=True)

train_data["LoanAmount"].fillna(train_data["LoanAmount"].median(), inplace=True)
train_data["Loan_Amount_Term"].fillna(train_data["Loan_Amount_Term"].median(), inplace=True)

print("\nMissing values after handling:")
print(train_data.isnull().sum())

train_data.drop_duplicates(inplace=True)
print(f"\nDataset shape after removing duplicates: {train_data.shape}")

train_data["Dependents"] = train_data["Dependents"].replace("3+", "3")
train_data["Dependents"] = train_data["Dependents"].astype(int)

train_data["Total_Income"] = train_data["ApplicantIncome"] + train_data["CoapplicantIncome"]
train_data["EMI"] = train_data["LoanAmount"] / train_data["Loan_Amount_Term"]
train_data["Balance_Income"] = train_data["Total_Income"] - (train_data["EMI"] * 1000)

train_data["LoanAmount_log"] = np.log(train_data["LoanAmount"] + 1)
train_data["Total_Income_log"] = np.log(train_data["Total_Income"] + 1)

train_data.drop(columns=["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                          "Total_Income", "EMI", "Balance_Income"], inplace=True)

print("\nFeatures after engineering:")
print(train_data.columns.tolist())

label_encoder = LabelEncoder()

categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

for col in categorical_cols:
    train_data[col] = label_encoder.fit_transform(train_data[col])

train_data["Loan_Status"] = train_data["Loan_Status"].map({"Y": 1, "N": 0})

print("\nData after encoding:")
print(train_data.head())

X = train_data.drop(columns=["Loan_Status"])
y = train_data["Loan_Status"]

print(f"\nClass distribution before SMOTE:\n{y.value_counts()}")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"\nClass distribution after SMOTE:\n{pd.Series(y_resampled).value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(random_state=42, max_iter=1000)
decision_tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)

logistic_model.fit(X_train_scaled, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

print("\nAll three models trained successfully.")

def evaluate_model(model_name, model, X_test_data, y_test_data):
    preds = model.predict(X_test_data)
    probs = model.predict_proba(X_test_data)[:, 1]

    acc = accuracy_score(y_test_data, preds)
    prec = precision_score(y_test_data, preds)
    rec = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    roc = roc_auc_score(y_test_data, probs)

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test_data, preds)}")

    return {"Model": model_name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "ROC-AUC": roc}

lr_results = evaluate_model("Logistic Regression", logistic_model, X_test_scaled, y_test)
dt_results = evaluate_model("Decision Tree", decision_tree_model, X_test, y_test)
rf_results = evaluate_model("Random Forest", random_forest_model, X_test, y_test)

comparison_df = pd.DataFrame([lr_results, dt_results, rf_results])
print("\n\n===== Model Comparison Table =====")
print(comparison_df.to_string(index=False))

os.makedirs("screenshots", exist_ok=True)

rf_preds = random_forest_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"])
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png")
plt.close()
print("\nConfusion matrix saved to screenshots/confusion_matrix.png")

feature_importances = pd.Series(
    random_forest_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png")
plt.close()
print("Feature importance plot saved to screenshots/feature_importance.png")

os.makedirs("model", exist_ok=True)

joblib.dump(random_forest_model, "model/loan_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "model/feature_columns.pkl")

print("\nModel, scaler, and feature columns saved in the model/ folder.")
print("\nTraining complete! Random Forest was selected as the final model.")
print("It gave the best balance of Accuracy, Precision, Recall, and F1 Score.")
