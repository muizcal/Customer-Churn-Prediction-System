import streamlit as st
import requests

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.number_input("SeniorCitizen", 0, 1)
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", 0, 72)
MonthlyCharges = st.number_input("MonthlyCharges", 0.0)
TotalCharges = st.number_input("TotalCharges", 0.0)

if st.button("Predict"):
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    st.write(response.json())
