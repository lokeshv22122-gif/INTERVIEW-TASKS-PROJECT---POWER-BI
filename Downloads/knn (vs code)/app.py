
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Delinquency Predictor", page_icon="💳", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

st.title("Loan Delinquency Prediction App")
st.write("Predict whether a loan applicant is likely to be **seriously delinquent (Sdelinquent)** using the trained KNN model.")

st.subheader("Enter applicant details")

term = st.selectbox("Loan Term", label_encoders["term"].classes_)
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
purpose = st.selectbox("Loan Purpose", label_encoders["purpose"].classes_)
home_ownership = st.selectbox("Home Ownership", label_encoders["home_ownership"].classes_)
age = st.selectbox("Age Group", label_encoders["age"].classes_)
fico = st.selectbox("FICO Range", label_encoders["FICO"].classes_)

if st.button("Predict"):
    input_df = pd.DataFrame({
        "term": [label_encoders["term"].transform([term])[0]],
        "gender": [label_encoders["gender"].transform([gender])[0]],
        "purpose": [label_encoders["purpose"].transform([purpose])[0]],
        "home_ownership": [label_encoders["home_ownership"].transform([home_ownership])[0]],
        "age": [label_encoders["age"].transform([age])[0]],
        "FICO": [label_encoders["FICO"].transform([fico])[0]],
    })

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]

    if pred == 1:
        st.error("Prediction: Likely Serious Delinquent")
    else:
        st.success("Prediction: Not Serious Delinquent")

    st.write(f"Probability of Not Serious Delinquent: **{pred_proba[0]:.2%}**")
    st.write(f"Probability of Serious Delinquent: **{pred_proba[1]:.2%}**")
