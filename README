# **Customer Churn Prediction System**

### This project is a Customer Churn Prediction System built using AI/ML techniques. It predicts whether a customer is likely to leave (churn) based on their account information and usage patterns. The system consists of:

- Backend: FastAPI application serving predictions via an API.

- Frontend: Streamlit app for interactive input and real-time predictions.

- Model: Random Forest Classifier trained on real customer data.

## Problem It Solves

Customer churn is a major issue for companies in telecom, banking, subscription services, and more. Losing customers means lost revenue.

This system helps businesses:

- Identify at-risk customers before they leave.

- Make data-driven retention decisions.

- Reduce costs of acquiring new customers by focusing on retention.

## Backend (FastAPI)

### Install dependencies
<pre> pip install -r requirements.txt </pre>

## Run the API
<pre> cd app
uvicorn main:app --reload
</pre>
The API will be available at http://127.0.0.1:8000.

## Frontend (Streamlit)
Run the frontend
<pre> cd frontend
streamlit run app.py
</pre>

Open the link provided in the terminal to access the web app

## Usage

- Input customer data in the frontend form.

- Click Predict.

- The app will return:

- Churn: Predicted class (Yes / No)

- Probability: Churn probability

## Training a New Model

If you want to retrain the model:
<pre> cd model
python train.py
</pre>
This will save a new model to saved_models/churn_model.pkl

## Notes

- Make sure the API is running before using the frontend.

- Only the following columns are used for prediction: ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "MonthlyCharges", "TotalCharges"]

## Docker setup is optional for this project
