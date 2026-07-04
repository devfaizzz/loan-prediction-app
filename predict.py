import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "loan_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "model", "feature_columns.pkl"))

def preprocess_input(gender, married, dependents, education,
                     self_employed, applicant_income, coapplicant_income,
                     loan_amount, loan_amount_term, credit_history, property_area):

    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Yes" else 0
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_area_encoded = property_area_map[property_area]

    total_income = applicant_income + coapplicant_income
    emi = loan_amount / loan_amount_term
    balance_income = total_income - (emi * 1000)

    loan_amount_log = np.log(loan_amount + 1)
    total_income_log = np.log(total_income + 1)

    input_data = {
        "Gender": gender_encoded,
        "Married": married_encoded,
        "Dependents": dependents,
        "Education": education_encoded,
        "Self_Employed": self_employed_encoded,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area_encoded,
        "LoanAmount_log": loan_amount_log,
        "Total_Income_log": total_income_log
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    return input_df

def predict_loan(gender, married, dependents, education,
                 self_employed, applicant_income, coapplicant_income,
                 loan_amount, loan_amount_term, credit_history, property_area):

    input_df = preprocess_input(
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result = "Approved" if prediction == 1 else "Rejected"
    confidence = round(max(probability) * 100, 2)

    return result, confidence, probability

if __name__ == "__main__":
    result, confidence, probs = predict_loan(
        gender="Male",
        married="Yes",
        dependents=0,
        education="Graduate",
        self_employed="No",
        applicant_income=5000,
        coapplicant_income=2000,
        loan_amount=150,
        loan_amount_term=360,
        credit_history=1,
        property_area="Urban"
    )
    print(f"Loan Status : {result}")
    print(f"Confidence  : {confidence}%")
    print(f"Probabilities -> Rejected: {probs[0]:.4f}, Approved: {probs[1]:.4f}")
