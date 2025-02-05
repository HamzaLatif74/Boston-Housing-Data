import streamlit as st
import joblib
import numpy as np

# Load the StandardScaler and Regression Model
scaler = joblib.load("scaling.pkl")   # Load the StandardScaler
model = joblib.load("regmodel.pkl")   # Load the trained Regression Model

# Streamlit UI
st.title("🏠 Boston House Price Prediction")
st.write("Enter the details below to predict the house price.")

# User Inputs for the 13 Features
CRIM = st.number_input("Per capita crime rate (CRIM)", min_value=0.0, value=0.1, step=0.1)
ZN = st.number_input("Proportion of land zoned (ZN)", min_value=0.0, value=10.0, step=1.0)
INDUS = st.number_input("Proportion of non-retail business acres (INDUS)", min_value=0.0, value=5.0, step=0.1)
CHAS = st.selectbox("Bounds Charles River (CHAS)", [0, 1])  # Binary input
NOX = st.number_input("Nitric oxide concentration (NOX)", min_value=0.0, value=0.5, step=0.01)
RM = st.number_input("Average rooms per dwelling (RM)", min_value=1.0, value=6.0, step=0.1)
AGE = st.number_input("Proportion of older owner-occupied units (AGE)", min_value=0.0, value=60.0, step=1.0)
DIS = st.number_input("Distance to employment centers (DIS)", min_value=0.0, value=4.0, step=0.1)
RAD = st.number_input("Accessibility to radial highways (RAD)", min_value=1, value=4, step=1)
TAX = st.number_input("Property-tax rate (TAX)", min_value=0, value=300, step=10)
PTRATIO = st.number_input("Pupil-teacher ratio (PTRATIO)", min_value=0.0, value=18.0, step=0.1)
B = st.number_input("Proportion of black residents (B)", min_value=0.0, value=350.0, step=1.0)
LSTAT = st.number_input("Lower status of the population (LSTAT)", min_value=0.0, value=12.0, step=0.1)

# Convert inputs to NumPy array
features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

# Predict Button
if st.button("🔮 Predict House Price"):
    # Scale input data
    features_scaled = scaler.transform(features)

    # Make Prediction
    predicted_price = model.predict(features_scaled)[0]

    # Show result
    st.success(f"🏡 Estimated House Price: **${predicted_price * 1000:.2f}**")

