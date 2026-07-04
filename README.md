# Loan Approval Prediction System

## 🏦 Project Overview

This project is an **AIML Internship Capstone Project** that builds a complete **Loan Approval Prediction System** using Machine Learning. The system predicts whether a loan application should be **Approved** or **Rejected** based on applicant details such as income, credit history, education, and property area.

The project follows the complete ML lifecycle — from data loading and preprocessing to model training, evaluation, and deployment via a Streamlit web application.

---

## 🎯 Problem Statement

Banks and financial institutions receive thousands of loan applications every day. Manually reviewing each application is time-consuming and prone to human error. The goal of this project is to build an automated ML-based system that can predict loan approval decisions based on historical data, helping banks make faster and more accurate decisions.

---

## 📌 Objectives

- Understand and explore the Kaggle Loan Prediction Dataset
- Perform data cleaning, feature engineering, and preprocessing
- Handle class imbalance using SMOTE
- Train and compare multiple classification models
- Select the best model based on evaluation metrics
- Deploy the model as an interactive Streamlit web application

---

## ✨ Features

- 🔍 **Loan Approval Prediction** based on 11 applicant features
- 📊 **Confidence Score** and **Prediction Probability** shown for each prediction
- 📋 **Application Summary** displayed after prediction
- 💡 **Actionable Tips** shown for rejected applications
- 🎨 **Premium Dark UI** with glassmorphism design
- ✅ **Input Validation** for all form fields

---

## 🛠️ Technology Stack

| Category       | Technology              |
|----------------|-------------------------|
| Language       | Python 3.10+            |
| ML Library     | Scikit-learn            |
| Data Analysis  | Pandas, NumPy           |
| Visualization  | Matplotlib, Seaborn     |
| Balancing      | Imbalanced-learn (SMOTE)|
| Model Saving   | Joblib                  |
| Deployment     | Streamlit               |

---

## 📂 Dataset

- **Source**: Kaggle – Loan Prediction Problem Dataset
- **Training File**: `dataset/train.csv` (614 rows, 13 columns)
- **Testing File**: `dataset/test.csv` (367 rows, 12 columns)
- **Target Column**: `Loan_Status` (Y = Approved, N = Rejected)

> The `test.csv` file does NOT contain the target column. It is used for final predictions only. All model training and evaluation is done using `train.csv`.

---

## 📁 Folder Structure

```
LoanApprovalPrediction/
│
├── dataset/
│   ├── train.csv              # Training dataset with labels
│   └── test.csv               # Test dataset without labels
│
├── notebook/
│   └── loan_analysis.ipynb    # Complete Jupyter Notebook
│
├── model/
│   ├── loan_model.pkl         # Saved trained model (Random Forest)
│   ├── scaler.pkl             # Saved StandardScaler
│   └── feature_columns.pkl    # Saved feature column order
│
├── app/
│   └── streamlit_app.py       # Streamlit web application
│
├── screenshots/
│   ├── confusion_matrix.png   # Confusion matrix plot
│   └── feature_importance.png # Feature importance plot
│
├── train.py                   # Model training script
├── predict.py                 # Prediction helper module
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

---

## ⚙️ Installation & Setup

### Step 1: Clone or Download the Project

If downloaded as ZIP, extract it to a folder of your choice.

### Step 2: Open Terminal / Command Prompt

Navigate to the project folder:

```bash
cd path/to/LoanApprovalPrediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 5: Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Train the Model

Run the training script first. This will:
- Load and preprocess the dataset
- Apply SMOTE for class balancing
- Train Logistic Regression, Decision Tree, and Random Forest
- Evaluate all models and print a comparison table
- Save the best model to the `model/` folder

```bash
python train.py
```

### Step 2: Test Predictions (Optional)

Run the prediction script to test a sample prediction:

```bash
python predict.py
```

### Step 3: Launch the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

This will open the web application at: `http://localhost:8501`

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~79%     | ~78%      | ~81%   | ~79%     | ~86%    |
| Decision Tree       | ~78%     | ~77%      | ~80%   | ~78%     | ~78%    |
| Random Forest       | ~82%     | ~82%      | ~83%   | ~82%     | ~90%    |

**Random Forest was selected as the final model** due to its highest accuracy, best ROC-AUC score, and balanced precision-recall trade-off.

---

## 🧪 How to Use the Application

1. Open the Streamlit app in your browser
2. Fill in the applicant details in the form:
   - Gender, Marital Status, Dependents, Education, Self-Employment
   - Applicant Income, Co-applicant Income, Loan Amount
   - Loan Term, Credit History, Property Area
3. Click **"Predict Loan Status"**
4. View the result: **Approved ✅** or **Rejected ❌**
5. Check the confidence score and prediction probabilities
6. Read the application summary and improvement tips

---

## 🔮 Future Scope

- Add more features like existing loan count and debt-to-income ratio
- Experiment with XGBoost or LightGBM for better accuracy
- Integrate with a real bank database using SQL
- Add user authentication for bank officer login
- Generate downloadable PDF reports for each prediction
- Deploy on cloud platforms (Heroku, AWS, or Render)

---

## 📄 License

This project is submitted as part of the **AIML Summer Internship Capstone Project** and is intended for educational purposes only.

---

## 👤 Author

**[Your Name Here]**
B.Tech CSE – 3rd Year
AIML Summer Internship, 2024

> *This project was developed as part of an internship capstone to demonstrate a complete machine learning development workflow.*
