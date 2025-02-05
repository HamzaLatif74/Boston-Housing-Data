import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the StandardScaler and Regression Model
scaler = joblib.load("scaling.pkl")   # Load StandardScaler
model = joblib.load("regmodel.pkl")   # Load trained Regression Model

# Define feature names
feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

# Streamlit UI
st.title("🏠 Boston House Price Prediction")
st.write("Enter the details below to predict the house price.")

# Create a two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    CRIM = st.number_input("CRIM (Crime Rate)", min_value=0.0, value=0.1, step=0.1)
    ZN = st.number_input("ZN (Zoned Land %)", min_value=0.0, value=10.0, step=1.0)
    INDUS = st.number_input("INDUS (Non-Retail Acres)", min_value=0.0, value=5.0, step=0.1)
    CHAS = st.selectbox("CHAS (Near River)", [0, 1])  # Binary input
    NOX = st.number_input("NOX (Pollution Level)", min_value=0.0, value=0.5, step=0.01)
    RM = st.number_input("RM (Rooms Per House)", min_value=1.0, value=6.0, step=0.1)
    AGE = st.number_input("AGE (Old Houses %)", min_value=0.0, value=60.0, step=1.0)

with col2:
    DIS = st.number_input("DIS (Distance to Work)", min_value=0.0, value=4.0, step=0.1)
    RAD = st.number_input("RAD (Highway Access)", min_value=1, value=4, step=1)
    TAX = st.number_input("TAX (Property Tax)", min_value=0, value=300, step=10)
    PTRATIO = st.number_input("PTRATIO (Teacher-Student Ratio)", min_value=0.0, value=18.0, step=0.1)
    B = st.number_input("B (Black Population %)", min_value=0.0, value=350.0, step=1.0)
    LSTAT = st.number_input("LSTAT (Lower Income %)", min_value=0.0, value=12.0, step=0.1)

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

    # Show Feature Importance
    st.subheader("📊 Feature Importance")

    # Get feature importance (only if model supports it)
    if hasattr(model, "coef_"):  # Works for LinearRegression
        importance = model.coef_[0]
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feature_names, importance, color="skyblue")
        ax.set_xlabel("Feature Weight")
        ax.set_title("Feature Importance in House Price Prediction")
        st.pyplot(fig)
    else:
        st.write("⚠️ Feature importance is not available for this model.")
