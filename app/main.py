from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np


with open("saved_models/churn_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    encoders = data["encoders"]
    columns = data["columns"]  

app = FastAPI(title="Customer Churn Prediction API")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(customer: Customer):
  
    df = pd.DataFrame([customer.dict()])


    for col in columns:
        if col not in df.columns:
            
            if col in encoders:
                df[col] = encoders[col].classes_[0]
            else:
                df[col] = 0


    df = df[columns]


    for col, encoder in encoders.items():
        if col in df.columns:
            # Replace unseen categories with first class
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            df[col] = encoder.transform(df[col])


    df_scaled = scaler.transform(df)

   
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]


    churn_label = encoders["Churn"].inverse_transform([pred])[0]

    return {"Churn": churn_label, "Probability": float(prob)}
