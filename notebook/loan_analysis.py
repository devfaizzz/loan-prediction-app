# ============================================================
# Loan Approval Prediction - Complete Analysis Notebook
# AIML Internship Capstone Project
# ============================================================

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
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

os.makedirs("../screenshots", exist_ok=True)

# ============================================================
# SECTION 1: Load Data
# WHY: We load train.csv for training. test.csv has no labels
#      so it's only used for final predictions via the app.
# ============================================================

print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

train_df = pd.read_csv("../dataset/train.csv")
test_df  = pd.read_csv("../dataset/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")
print("\nFirst 5 rows of train data:")
print(train_df.head())

# ============================================================
# SECTION 2: Dataset Understanding
# WHY: Before doing anything, we must understand what each
#      column means so we know how to clean it properly.
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Dataset Understanding")
print("=" * 60)

print("\nColumn Data Types:")
print(train_df.dtypes)

print("\nBasic Statistics (Numerical Columns):")
print(train_df.describe())

print("\nUnique values in each column:")
for col in train_df.columns:
    print(f"  {col}: {train_df[col].nunique()} unique -> {train_df[col].unique()[:5]}")

# ============================================================
# SECTION 3: Missing Values
# WHY: Most real datasets have missing values. If we don't
#      handle them, the ML model will throw errors or give
#      wrong predictions.
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Missing Value Analysis")
print("=" * 60)

missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df)) * 100
missing_df = pd.DataFrame({"Missing Count": missing, "Percentage": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])

# Visualize missing values
plt.figure(figsize=(10, 5))
sns.heatmap(train_df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap (Yellow = Missing)")
plt.tight_layout()
plt.savefig("../screenshots/missing_values_heatmap.png")
plt.show()
print("Saved: missing_values_heatmap.png")

# ============================================================
# SECTION 4: EDA - Exploratory Data Analysis
# WHY: EDA helps us understand patterns, distributions, and
#      relationships in the data before modeling.
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Exploratory Data Analysis (EDA)")
print("=" * 60)

# 4A: Target Variable Distribution
plt.figure(figsize=(6, 4))
train_df["Loan_Status"].value_counts().plot(kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black")
plt.title("Loan Status Distribution (Y=Approved, N=Rejected)")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../screenshots/loan_status_distribution.png")
plt.show()
print("Class distribution:")
print(train_df["Loan_Status"].value_counts())

# 4B: Applicant Income Distribution
plt.figure(figsize=(8, 4))
sns.histplot(train_df["ApplicantIncome"], bins=40, kde=True, color="#3498db")
plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.tight_layout()
plt.savefig("../screenshots/applicant_income_distribution.png")
plt.show()

# 4C: Loan Amount Distribution
plt.figure(figsize=(8, 4))
sns.histplot(train_df["LoanAmount"].dropna(), bins=40, kde=True, color="#9b59b6")
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount (in thousands)")
plt.tight_layout()
plt.savefig("../screenshots/loan_amount_distribution.png")
plt.show()

# 4D: Credit History Distribution
plt.figure(figsize=(6, 4))
train_df["Credit_History"].value_counts().plot(kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black")
plt.title("Credit History Distribution (1=Good, 0=Bad)")
plt.xlabel("Credit History")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../screenshots/credit_history_distribution.png")
plt.show()

# 4E: Property Area Distribution
plt.figure(figsize=(6, 4))
train_df["Property_Area"].value_counts().plot(kind="bar", color=["#f39c12", "#27ae60", "#2980b9"], edgecolor="black")
plt.title("Property Area Distribution")
plt.xlabel("Property Area")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../screenshots/property_area_distribution.png")
plt.show()

# 4F: Loan Status vs Credit History (very important feature!)
plt.figure(figsize=(7, 5))
pd.crosstab(train_df["Credit_History"], train_df["Loan_Status"]).plot(
    kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black"
)
plt.title("Credit History vs Loan Status")
plt.xlabel("Credit History (0=Bad, 1=Good)")
plt.xticks(rotation=0)
plt.legend(["Rejected", "Approved"])
plt.tight_layout()
plt.savefig("../screenshots/credit_history_vs_loan_status.png")
plt.show()

# 4G: Loan Amount vs Applicant Income Scatter
plt.figure(figsize=(8, 5))
colors = train_df["Loan_Status"].map({"Y": "#2ecc71", "N": "#e74c3c"})
plt.scatter(train_df["ApplicantIncome"], train_df["LoanAmount"], c=colors, alpha=0.6)
plt.title("Applicant Income vs Loan Amount (Green=Approved, Red=Rejected)")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.savefig("../screenshots/income_vs_loanamount_scatter.png")
plt.show()

# 4H: Boxplot of Loan Amount by Loan Status
plt.figure(figsize=(7, 5))
sns.boxplot(x="Loan_Status", y="LoanAmount", data=train_df, palette={"Y": "#2ecc71", "N": "#e74c3c"})
plt.title("Loan Amount Distribution by Approval Status")
plt.tight_layout()
plt.savefig("../screenshots/loanamount_boxplot.png")
plt.show()

# 4I: Education vs Loan Status
plt.figure(figsize=(7, 5))
pd.crosstab(train_df["Education"], train_df["Loan_Status"]).plot(
    kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black"
)
plt.title("Education vs Loan Status")
plt.xticks(rotation=0)
plt.legend(["Rejected", "Approved"])
plt.tight_layout()
plt.savefig("../screenshots/education_vs_loan_status.png")
plt.show()

print("All EDA plots saved to screenshots/ folder.")

# ============================================================
# SECTION 5: Data Preprocessing
# WHY: Raw data cannot be directly fed to ML models. We need
#      to handle missing values, fix data types, and clean it.
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Data Preprocessing")
print("=" * 60)

df = train_df.copy()
df.drop(columns=["Loan_ID"], inplace=True)

# Fill missing categorical with mode
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# Fill missing numerical with median
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)

print("Missing values after handling:")
print(df.isnull().sum())

# Fix Dependents column (3+ -> 3)
df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

# Remove duplicates
df.drop_duplicates(inplace=True)
print(f"\nShape after preprocessing: {df.shape}")

# ============================================================
# SECTION 6: Feature Engineering
# WHY: We create new meaningful features from existing ones
#      that help the model learn better patterns.
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Feature Engineering")
print("=" * 60)

df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["EMI"]          = df["LoanAmount"] / df["Loan_Amount_Term"]
df["Balance_Income"] = df["Total_Income"] - (df["EMI"] * 1000)

# Log transform reduces right skewness in income & loan amount
df["LoanAmount_log"]   = np.log(df["LoanAmount"] + 1)
df["Total_Income_log"] = np.log(df["Total_Income"] + 1)

df.drop(columns=["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                  "Total_Income", "EMI", "Balance_Income"], inplace=True)

print("Features after engineering:")
print(df.columns.tolist())

# Visualize log-transformed income distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Total_Income_log"], kde=True, ax=axes[0], color="#3498db")
axes[0].set_title("Total Income (Log Transformed)")
sns.histplot(df["LoanAmount_log"], kde=True, ax=axes[1], color="#9b59b6")
axes[1].set_title("Loan Amount (Log Transformed)")
plt.tight_layout()
plt.savefig("../screenshots/log_transformed_distributions.png")
plt.show()

# ============================================================
# SECTION 7: Encoding Categorical Variables
# WHY: ML models work with numbers only. We convert text
#      categories to numbers using LabelEncoder.
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Encoding Categorical Variables")
print("=" * 60)

le = LabelEncoder()
categorical_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

print("Data after encoding (first 3 rows):")
print(df.head(3))

# ============================================================
# SECTION 8: Correlation Analysis
# WHY: Correlation heatmap shows which features are strongly
#      related to the target. Helps in feature selection.
# ============================================================

plt.figure(figsize=(11, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../screenshots/correlation_heatmap.png")
plt.show()
print("Correlation heatmap saved.")

# ============================================================
# SECTION 9: Class Imbalance & SMOTE
# WHY: If one class (Approved) has more samples than the other
#      (Rejected), the model will be biased. SMOTE creates
#      synthetic samples for the minority class.
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Handling Class Imbalance with SMOTE")
print("=" * 60)

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

print(f"Before SMOTE: {y.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"After  SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

# Visualize before/after
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
y.value_counts().plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="black")
axes[0].set_title("Before SMOTE")
axes[0].set_xticklabels(["Approved", "Rejected"], rotation=0)
pd.Series(y_res).value_counts().plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"], edgecolor="black")
axes[1].set_title("After SMOTE (Balanced)")
axes[1].set_xticklabels(["Approved", "Rejected"], rotation=0)
plt.tight_layout()
plt.savefig("../screenshots/smote_before_after.png")
plt.show()

# ============================================================
# SECTION 10: Train/Test Split and Scaling
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Train-Test Split and Feature Scaling")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ============================================================
# SECTION 11: Model Training
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Training Models")
print("=" * 60)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

lr_model.fit(X_train_sc, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print("All 3 models trained!")

# ============================================================
# SECTION 12: Model Evaluation
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: Model Evaluation")
print("=" * 60)

def get_metrics(name, model, X_data, y_data):
    preds = model.predict(X_data)
    probs = model.predict_proba(X_data)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_data, preds), 4),
        "Precision": round(precision_score(y_data, preds), 4),
        "Recall":    round(recall_score(y_data, preds), 4),
        "F1 Score":  round(f1_score(y_data, preds), 4),
        "ROC-AUC":   round(roc_auc_score(y_data, probs), 4),
    }

results = [
    get_metrics("Logistic Regression", lr_model, X_test_sc, y_test),
    get_metrics("Decision Tree",       dt_model, X_test,    y_test),
    get_metrics("Random Forest",       rf_model, X_test,    y_test),
]

results_df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(results_df.to_string(index=False))

# Bar chart comparison
results_df.set_index("Model")[["Accuracy", "F1 Score", "ROC-AUC"]].plot(
    kind="bar", figsize=(10, 5), edgecolor="black"
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0.6, 1.0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../screenshots/model_comparison.png")
plt.show()

# Confusion Matrix for Random Forest
rf_preds = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../screenshots/confusion_matrix.png")
plt.show()

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds, target_names=["Rejected", "Approved"]))

# Feature Importance
fi = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=fi.values, y=fi.index, palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("../screenshots/feature_importance.png")
plt.show()

# ============================================================
# SECTION 13: Save Model
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: Saving Model, Scaler, and Feature Columns")
print("=" * 60)

os.makedirs("../model", exist_ok=True)
joblib.dump(rf_model, "../model/loan_model.pkl")
joblib.dump(scaler,   "../model/scaler.pkl")
joblib.dump(X.columns.tolist(), "../model/feature_columns.pkl")

print("Saved: model/loan_model.pkl")
print("Saved: model/scaler.pkl")
print("Saved: model/feature_columns.pkl")
print("\nNotebook analysis complete!")
