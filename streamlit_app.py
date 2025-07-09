import streamlit as st
import pandas as pd
import pickle

# Load saved objects
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder_columns = pickle.load(open('encoder_columns.pkl', 'rb'))

# Streamlit UI config
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>🌾 Crop Yield Prediction Per Country</h1>", unsafe_allow_html=True)


# Input form
with st.form("prediction_form"):
    year = st.text_input("Year")
    rainfall = st.text_input("Average Rainfall (mm/year)")
    pesticides = st.text_input("Pesticides Used (tonnes)")
    temperature = st.text_input("Average Temperature (°C)")

    area = st.selectbox("Country", ["Albania", "India", "Brazil", "China"])
    crop = st.selectbox("Crop Type", ["Maize", "Wheat", "Rice, paddy", "Barley"])

    submitted = st.form_submit_button("Predict Yield")

# On form submission
if submitted:
    try:
        # Convert inputs to correct data types
        year = int(year)
        rainfall = float(rainfall)
        pesticides = float(pesticides)
        temperature = float(temperature)

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Year': year,
            'average_rain_fall_mm_per_year': rainfall,
            'pesticides_tonnes': pesticides,
            'avg_temp': temperature,
            'Area': area,
            'Item': crop
        }])

        # One-hot encode and align
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=encoder_columns, fill_value=0)

        # Scale features
        input_scaled = scaler.transform(input_encoded)

        # Make prediction
        prediction = rf_model.predict(input_scaled)[0]

        # Show result
        st.success(f"🌱 **Predicted Crop Yield**: {prediction:.2f} hg/ha")
    
    except ValueError:
        st.error("Please enter valid numeric values for year, rainfall, pesticides, and temperature.")
