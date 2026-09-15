import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Rain Prediction",
    layout="centered"
)

# -----------------------------
# Load Trained Model
# -----------------------------
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Title
# -----------------------------
st.title("Rain Prediction System")
st.write("Enter the weather details below to predict whether it will rain.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
temperature = st.number_input(
    "Temperature",
    min_value=-50.0,
    max_value=60.0,
    value=30.0
)

cloud_cover = st.number_input(
    "Cloud Cover",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

wind_speed = st.number_input(
    "Wind Speed",
    min_value=0.0,
    max_value=200.0,
    value=2.0
)

humidity = st.number_input(
    "Humidity",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

pressure = st.number_input(
    "Atmospheric Pressure",
    min_value=800.0,
    max_value=1200.0,
    value=1020.0
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Rain"):

    input_data = np.array([[
        temperature,
        cloud_cover,
        wind_speed,
        humidity,
        pressure
    ]])

    prediction = model.predict(input_data)[0]

    # Prediction result
    if prediction == 1:
        st.error("Prediction: Rain")
    else:
        st.success("Prediction: No Rain")

    # Prediction probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        no_rain_probability = probabilities[0] * 100
        rain_probability = probabilities[1] * 100

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        col1.metric(
            "No Rain",
            f"{no_rain_probability:.2f}%"
        )

        col2.metric(
            "Rain",
            f"{rain_probability:.2f}%"
        )