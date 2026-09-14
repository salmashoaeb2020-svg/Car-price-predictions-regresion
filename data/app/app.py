import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Load Model
# ==========================================

model_path = "../model/car_price_model.pkl"

model = joblib.load(model_path)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("🚗 Egyptian Car Price Predictor")

st.write(
    "Enter the car specifications below "
    "to estimate its price in Egyptian Pounds (EGP)."
)


# ==========================================
# User Inputs
# ==========================================

brand = st.selectbox(
    "Car Brand",
    [
        "Toyota",
        "Hyundai",
        "Kia",
        "BMW",
        "Mercedes",
        "Nissan"
    ]
)


# Model depends on Brand

models = {
    "Toyota": ["Corolla", "Camry"],
    "Hyundai": ["Elantra", "Tucson"],
    "Kia": ["Sportage", "Cerato"],
    "BMW": ["320i", "X3"],
    "Mercedes": ["C180"],
    "Nissan": ["Sunny"]
}


car_model = st.selectbox(
    "Car Model / Line",
    models[brand]
)


year = st.number_input(
    "Model Year",
    min_value=1990,
    max_value=2026,
    value=2022,
    step=1
)


kilometers = st.number_input(
    "Kilometers",
    min_value=0,
    max_value=1000000,
    value=30000,
    step=1000
)


condition = st.selectbox(
    "Condition",
    [
        "Zero",
        "Almost New",
        "Used"
    ]
)


trim = st.selectbox(
    "Trim / Category",
    [
        "First",
        "Second",
        "Medium"
    ]
)


# ==========================================
# Image Upload
# ==========================================

st.subheader("📷 Car Image")

uploaded_image = st.file_uploader(
    "Upload a picture of the car",
    type=["jpg", "jpeg", "png"]
)


if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="Uploaded Car",
        use_container_width=True
    )


# ==========================================
# Prediction Button
# ==========================================

if st.button("Predict Price", type="primary"):

    input_data = pd.DataFrame(
        {
            "Brand": [brand],
            "Model": [car_model],
            "Year": [year],
            "Kilometers": [kilometers],
            "Condition": [condition],
            "Trim": [trim]
        }
    )

    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success(
        f"Estimated Price: {predicted_price:,.0f} EGP"
    )
