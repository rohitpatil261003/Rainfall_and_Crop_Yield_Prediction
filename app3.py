import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Rainfall & Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading with Caching ---
@st.cache_resource
def load_models():
    rainfall_path = "rainfall_pred.pkl"
    crop_path = "crop_yield_model.pkl"

    rainfall_model = joblib.load(rainfall_path) if os.path.exists(rainfall_path) else None
    crop_model = joblib.load(crop_path) if os.path.exists(crop_path) else None

    return rainfall_model, crop_model

rainfall_pipeline, crop_pipeline = load_models()

# --- App Header ---
st.title("🌧️ AgriPredict AI 🌾")
st.markdown(
   "**Rainfall** Forecasting and **Crop Yield** Prediction Using Machine Learning."
)

# Verify model availability
if rainfall_pipeline is None or crop_pipeline is None:
    missing = []
    if rainfall_pipeline is None:
        missing.append("`rainfall_pred.pkl`")
    if crop_pipeline is None:
        missing.append("`crop_yield_model.pkl`")
    st.error(f"❌ Missing trained pipeline file(s): {', '.join(missing)}. Please ensure both `.pkl` files are placed in the app directory.")
    st.stop()

# --- Sidebar: Location & Crop Configuration ---
st.sidebar.header("📍 Region & Crop Selection")

states_list = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

selected_state = st.sidebar.selectbox("Select State 🌍", sorted(states_list), index=sorted(states_list).index("Tamil Nadu") if "Tamil Nadu" in states_list else 0)
selected_year = st.sidebar.number_input("Forecast Year 📅", min_value=1950, max_value=2050, value=2024, step=1)

selected_crop = st.sidebar.selectbox("Crop Type 🌽", ["Rice", "Sugarcane", "Soybean", "Pulses", "Millet", "Wheat", "Cotton", "Maize"])
selected_season = st.sidebar.selectbox("Cropping Season 🌦️", ["Rabi", "Kharif", "Annual", "Zaid"])
farm_area = st.sidebar.number_input("Farm Area (Hectares)🚜", min_value=0.1, max_value=5000.0, value=25.0, step=1.0)

# --- Section 1: Weather & Climate Inputs (For Rainfall Model) ---
st.subheader("1️⃣ Rainfall Prediction 🌦️")
col_w1, col_w2, col_w3 = st.columns(3)

with col_w1:
    temp_val = st.slider("Average Temperature (°C) 🌡️", min_value=5.0, max_value=45.0, value=24.5, step=0.1)
    humidity_val = st.slider("Relative Humidity (%) 💧", min_value=10.0, max_value=100.0, value=65.0, step=0.5)

with col_w2:
    pressure_val = st.number_input("Atmospheric Pressure (hPa)🌬️", min_value=950.0, max_value=1050.0, value=1010.5, step=0.1)
    wind_speed_val = st.slider("Avg Wind Speed (km/h) 💨", min_value=0.0, max_value=50.0, value=8.5, step=0.1)
    cloud_cover_val = st.slider("Cloud Cover (%) ☁️", min_value=0.0, max_value=100.0, value=55.0, step=1.0)

with col_w3:
    prev_year_rain = st.number_input("Previous Year Rainfall (mm)📈", min_value=0.0, max_value=6000.0, value=850.0, step=10.0)
    prev_3yr_rain = st.number_input("Previous 3-Year Avg Rainfall (mm)📈", min_value=0.0, max_value=6000.0, value=820.0, step=10.0)

# --- Section 2: Soil & Nutrient Characteristics (For Crop Yield Model) ---
st.subheader("2️⃣ Crop Yield Prediction 🌱")
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    soil_type = st.selectbox("Soil Type 🪴", ["Black", "Alluvial", "Red", "Clayey", "Sandy", "Loamy"])
    soil_ph = st.slider("Soil pH Level 🧪", min_value=4.0, max_value=9.5, value=6.6, step=0.1)

with col_s2:
    nitrogen = st.number_input("Nitrogen (N) 🅽- kg/ha", min_value=0.0, max_value=300.0, value=75.0, step=1.0)
    phosphorus = st.number_input("Phosphorus (P) 🅿️- kg/ha", min_value=0.0, max_value=200.0, value=40.0, step=1.0)

with col_s3:
    potassium = st.number_input("Potassium (K) 🅺- kg/ha", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    fertilizer_total = fertilizer_total = float(nitrogen + phosphorus + potassium)

st.markdown("---")

# --- Integrated Prediction Logic ---
if st.button("🚀 Run Prediction 🤖", type="primary", use_container_width=True):
    try:
        # STEP 1: Construct Input Payload for Rainfall Model
        rainfall_input = pd.DataFrame([{
            "Year": selected_year,
            "State": selected_state,
            "Avg_Temperature_C": float(temp_val),
            "Avg_Humidity_percent": float(humidity_val),
            "Avg_Pressure_hPa": float(pressure_val),
            "Avg_Wind_Speed_kmh": float(wind_speed_val),
            "Avg_Cloud_Cover_percent": float(cloud_cover_val),
            "Previous_Year_Rainfall_mm": float(prev_year_rain),
            "Previous_3_Year_Avg_Rainfall_mm": float(prev_3yr_rain)
        }])

        predicted_rainfall = float(rainfall_pipeline.predict(rainfall_input)[0])
        predicted_rainfall = max(0.0, predicted_rainfall)

        # STEP 2: Construct Input Payload for Crop Yield Model (Injecting Predicted Rainfall)
        crop_input = pd.DataFrame([{
            "Crop": selected_crop,
            "State": selected_state,
            "Season": selected_season,
            "Area_hectares": float(farm_area),
            "Annual_Rainfall_mm": predicted_rainfall,  # Automatically passed from Step 1!
            "Avg_Temperature_C": float(temp_val),
            "Humidity_percent": float(humidity_val),
            "Soil_Type": soil_type,
            "Soil_pH": float(soil_ph),
            "N_kg_per_ha": float(nitrogen),
            "P_kg_per_ha": float(phosphorus),
            "K_kg_per_ha": float(potassium),
            "Fertilizer_kg_per_ha": float(fertilizer_total)
        }])

        predicted_yield = float(crop_pipeline.predict(crop_input)[0])
        predicted_yield = max(0.0, predicted_yield)
        total_production = predicted_yield * farm_area

        # --- Display Results ---
        st.success("✅ End-to-End Pipeline Executed Successfully!")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label=f"🌧️ Forecasted Annual Rainfall ({selected_year})",
                value=f"{predicted_rainfall:.1f} mm",
                delta=f"{predicted_rainfall - prev_year_rain:.1f} mm vs Prev Year"
            )

        with res_col2:
            st.metric(
                label=f"🌾 Estimated Yield ({selected_crop})",
                value=f"{predicted_yield:.3f} tonnes/ha"
            )

        with res_col3:
            st.metric(
                label="🚜 Estimated Total Production",
                value=f"{total_production:.2f} tonnes",
                help=f"Calculated as: {predicted_yield:.3f} tonnes/ha × {farm_area} hectares"
            )

        # Expandable inspection of transformed inputs
        with st.expander("🔍 View Connected Payloads"):
            st.write("**Payload Passed to Rainfall Model:**")
            st.dataframe(rainfall_input)
            st.write("**Payload Passed to Crop Yield Model (with Rainfall output piped in):**")
            st.dataframe(crop_input)

    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        st.info("Ensure both models were saved as complete `Pipeline` objects with their ColumnTransformers included.")