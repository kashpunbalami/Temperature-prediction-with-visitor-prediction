import streamlit as st
import pickle
import numpy as np
import time

# Load the trained models
with open("linear_regression.pkl", "rb") as file:
    weather_model = pickle.load(file)

with open("tourism.pkl", "rb") as file:
    tourism_model = pickle.load(file)

# Streamlit UI
st.title("🌤 Tourism Prediction App")

st.markdown("### Step 1: Predict Temperature")

# User inputs for weather prediction
slp = st.number_input("Enter Sea Level Pressure (slp) (range: 1011-1031):", min_value=1011.0, max_value=1031.0, value=1013.0)
dewpt = st.number_input("Enter Dew Point Temperature (dewpt) (range: 8.1-45.3):", min_value=8.1, max_value=45.3, value=10.0)
solar_rad = st.number_input("Enter Solar Radiation (solar_rad) (range: 0.0-484.1):", min_value=0.0, max_value=484.1, value=200.0)
rh = st.number_input("Enter Relative Humidity (rh) (range:28.0-100.0):", min_value=28.0, max_value=100.0, value=50.0)
pres = st.number_input("Enter Pressure (pres) (range:1003.0-1023.0):", min_value=1003.0, max_value=1023.0, value=1013.0)
app_temp = st.number_input("Enter Apparent Temperature (app_temp) (range:18.7-50.2):", min_value=18.7, max_value=50.2, value=25.0)

# Initialize session state for predicted temperature
if "predicted_temp" not in st.session_state:
    st.session_state.predicted_temp = None  # Store temperature after prediction

if st.button("Predict Temperature"):
    input_data = np.array([[slp, dewpt, solar_rad, rh, pres, app_temp]])
    st.session_state.predicted_temp = weather_model.predict(input_data)[0]  # Save to session state
    st.success(f"🌡️ Predicted Temperature: **{st.session_state.predicted_temp:.2f}°C**")
    time.sleep(2)  # Pause to simulate real-time processing

# Display predicted temperature if available
if st.session_state.predicted_temp is not None:
    st.markdown(f"**🌡️ Current Predicted Temperature: {st.session_state.predicted_temp:.2f}°C**")

# Proceed only if temperature is predicted
if st.session_state.predicted_temp is not None:
    st.title("👥 Visitor Prediction System Based on Predicted Temperature")

    # User inputs for tourism prediction
    resource_prediction = st.number_input("Enter Resource Prediction (range: 0.1843-0.9767):", min_value=0.1843, max_value=0.9767, value=0.1843)
    visitor_satisfaction = st.number_input("Enter Visitor Satisfaction (range:1.014-9.9977):", min_value=1.014, max_value=9.9977, value=7.0)
    resource_usage_rate = st.number_input("Enter Resource Usage Rate (range:0.3001-0.9979):", min_value=0.3001, max_value=0.9979, value=0.3001)

    # Keep displaying predicted temperature
    st.number_input(
        "Predicted Temperature:", 
        value=st.session_state.predicted_temp, 
        disabled=True
    )

    if st.button("Predict Visitor Count"):
        tourism_input = np.array([[resource_prediction, visitor_satisfaction, resource_usage_rate, st.session_state.predicted_temp]])
        predicted_visitors = tourism_model.predict(tourism_input)[0]
        st.session_state.predicted_visitors = int(predicted_visitors)  # Store visitor count

    # Display visitor count if predicted
    if "predicted_visitors" in st.session_state:
        st.success(f"👥 Predicted Visitor Count: **{st.session_state.predicted_visitors}**")
