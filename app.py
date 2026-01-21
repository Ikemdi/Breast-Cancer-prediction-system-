import streamlit as st
import joblib
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# --- Load Model and Scaler ---
# Using absolute paths to ensure it works across different deployment environments
MODEL_PATH = os.path.join('model', 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Please run model_building.py first.")

# --- UI Header ---
st.title("⚕️ Breast Cancer Prediction System")
st.markdown("""
This application uses a **Logistic Regression** model to predict whether a breast tumor is **Benign** or **Malignant** based on FNA image features.
""")

# --- Input Form ---
st.subheader("Input Tumor Features")
with st.container():
    # Creating two columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, value=14.1, help="Mean of distances from center to points on the perimeter")
        texture_mean = st.number_input("Texture Mean", min_value=0.0, value=19.3, help="Standard deviation of gray-scale values")
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=92.0)

    with col2:
        area_mean = st.number_input("Area Mean", min_value=0.0, value=654.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.1, format="%.4f")

# --- Prediction Logic ---
if st.button("Predict Diagnosis", type="primary"):
    # 1. Organize inputs into a 2D array
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
    
    # 2. Scale the input using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # 3. Perform Prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled) # Optional: shows confidence
    
    # 4. Display Results
    st.divider()
    if prediction[0] == 1:
        st.error(f"### Prediction: **Malignant**")
        st.write(f"Confidence Level: {prediction_proba[0][1]:.2%}")
    else:
        st.success(f"### Prediction: **Benign**")
        st.write(f"Confidence Level: {prediction_proba[0][0]:.2%}")

# --- Disclaimer (Mandatory per Project Instructions) ---
st.warning("**Disclaimer:** This system is strictly for educational purposes and must not be presented or used as a medical diagnostic tool.")