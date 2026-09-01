import streamlit as st
import numpy as np
import joblib
from flask import flask 
app = Flask(__name__)
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
transaction_id = st.text_input("Transaction ID")
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)
st.tittle("Fraud Detection Dashboard")
st.write("Welcome to the Fraud Detection Dashboard")

time = st.number_input(
    "Time",
    min_value=0.0,
    value=100.0
)

amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=100.0
)

feature1 = st.number_input(
    "Feature 1",
    value=0.0
)

feature2 = st.number_input(
    "Feature 2",
    value=0.0
)

if st.button("🔍 Check Transaction"):

    if transaction_id == "":
        st.warning("Please enter Transaction ID")

     app.run(debug=True)

        input_data = np.array([
            [time, amount, feature1, feature2]
        ])

        if scaler is not None:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data

        prediction = model.predict(input_scaled)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = 0

        if prediction == 1:

            st.error("🚨 FRAUD TRANSACTION DETECTED")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        else:

            st.success("✅ LEGITIMATE TRANSACTION")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )
