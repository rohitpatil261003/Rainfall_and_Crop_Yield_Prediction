import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists("crop_yield_model.pkl"):
        return joblib.load("crop_yield_model.pkl")
    return None

pipeline = load_model()

st.title("🌾 Crop Yield Prediction System")
st.write("Predict crop yield (tonnes per hectare) based on soil, climate, and crop parameters.")

if pipeline is None:
    st.error("Model file `crop_yield_model.pkl` not found! Please train and save the pipeline first.")
    st.stop()

# Form Inputs
col1, col2, col3 = st.columns(3)

with col1:
    crop = st.selectbox("Crop", ["Rice", "Sugarcane", "Soybean", "Pulses", "Millet", "Wheat", "Cotton", "Maize"])
    state = st.selectbox("State", ["Tamil Nadu", "Uttar Pradesh", "Andhra Pradesh", "Madhya Pradesh", "Karnataka", "Punjab", "Maharashtra", "Rajasthan"])
    season = st.selectbox("Season", ["Rabi", "Kharif", "Annual", "Zaid"])
    soil_type = st.selectbox("Soil Type", ["Black", "Alluvial", "Red", "Clayey", "Sandy", "Loamy"])

with col2:
    area = st.number_input("Area (hectares)", min_value=0.1, max_value=500.0, value=25.0, step=1.0)
    rainfall = st.number_input("Annual Rainfall (mm)", min_value=100.0, max_value=4000.0, value=950.0, step=25.0)
    temp = st.slider("Avg Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.5)
    humidity = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0, step=1.0)

with col3:
    soil_ph = st.slider("Soil pH", min_value=4.0, max_value=9.5, value=6.8, step=0.1)
    n_kg = st.number_input("Nitrogen (N) kg/ha", min_value=10.0, max_value=250.0, value=85.0, step=5.0)
    p_kg = st.number_input("Phosphorus (P) kg/ha", min_value=5.0, max_value=150.0, value=35.0, step=5.0)
    k_kg = st.number_input("Potassium (K) kg/ha", min_value=5.0, max_value=150.0, value=30.0, step=5.0)
    fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=10.0, max_value=500.0, value=120.0, step=10.0)

input_df = pd.DataFrame([{
    "Crop": crop,
    "State": state,
    "Season": season,
    "Area_hectares": float(area),
    "Annual_Rainfall_mm": float(rainfall),
    "Avg_Temperature_C": float(temp),
    "Humidity_percent": float(humidity),
    "Soil_Type": soil_type,
    "Soil_pH": float(soil_ph),
    "N_kg_per_ha": float(n_kg),
    "P_kg_per_ha": float(p_kg),
    "K_kg_per_ha": float(k_kg),
    "Fertilizer_kg_per_ha": float(fertilizer)
}])

st.markdown("---")
if st.button("🌱 Predict Yield", type="primary", use_container_width=True):
    try:
        pred_yield = pipeline.predict(input_df)[0]
        pred_yield = max(0.0, float(pred_yield))
        total_prod = pred_yield * area

        res1, res2 = st.columns(2)
        with res1:
            st.metric("Estimated Yield", f"{pred_yield:.3f} tonnes/ha")
        with res2:
            st.metric("Estimated Total Production", f"{total_prod:.2f} tonnes")
    except Exception as e:
        st.error(f"Prediction failed: {e}")