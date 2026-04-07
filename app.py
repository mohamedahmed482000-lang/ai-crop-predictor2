# ================= Prediction =================

import streamlit as st
import joblib
import pandas as pd

# ================= Load =================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

df = pd.read_csv("Crop_recommendation.csv")
target_column = 'crop' if 'crop' in df.columns else df.columns[-1]
X = df.drop(target_column, axis=1)

# ================= Page =================
st.set_page_config(page_title="Crop Recommender", page_icon="🌾", layout="wide")

# ================= Header =================
st.title("🌾 Smart Crop Recommendation System")
st.markdown("AI-powered system to recommend the best crop based on soil and weather conditions")

st.markdown("---")

# ================= Layout =================
col1, col2 = st.columns(2)

inputs = []

with col1:
    for col in X.columns[:len(X.columns)//2]:
        val = st.number_input(f"{col}", value=0.0)
        inputs.append(val)

with col2:
    for col in X.columns[len(X.columns)//2:]:
        val = st.number_input(f"{col}", value=0.0)
        inputs.append(val)

st.markdown("---")

# ================= Prediction =================

st.markdown("### 🔍 Get Recommendation")

predict_btn = st.button("🚀 Predict Crop", use_container_width=True)

if predict_btn:
    with st.spinner("Analyzing data... ⏳"):
        try:
            input_data = [inputs]
            input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)
            crop_name = encoder.inverse_transform(prediction)

            st.success(f"🌱 Recommended Crop: {crop_name[0]}")

            st.balloons()  # شكل حلو 😄

        except Exception as e:
            st.error(f"Error: {e}")
