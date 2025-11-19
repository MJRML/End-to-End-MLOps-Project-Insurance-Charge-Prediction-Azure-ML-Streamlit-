import streamlit as st
import requests
import json

# -----------------------------
# CONFIGURATION
# -----------------------------
ENDPOINT = "https://insurance-endpoint.westeurope.inference.ml.azure.com/score"

# Your primary key
API_KEY = "xxxx"

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Insurance Charge Predictor", page_icon="💰")

st.title("💰 Insurance Charge Prediction")
st.write("Enter the details below to estimate insurance charges.")

# -----------------------------
# INPUT FORM
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict"):
    payload = {
        "data": [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(ENDPOINT, json=payload, headers=headers)

    try:
        # Azure ML sometimes returns a STRING containing JSON → parse twice
        raw = response.json()

        # If raw is a string, load it as JSON
        if isinstance(raw, str):
            raw = json.loads(raw)

        predicted_value = raw["predictions"][0]

        st.success(f"Predicted Charges: **${predicted_value:,.2f}**")

    except Exception as e:
        st.error(" Error calling Azure ML endpoint")
        st.code(response.text)
        st.write(e)
