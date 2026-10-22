import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Annual Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Trained Pipeline ---
@st.cache_resource
def load_pipeline():
    model_paths = ["rainfall_pred.pkl", "rainfall_prediction_pipeline.pkl"]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

pipeline = load_pipeline()

# --- Custom Styling ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🌧️ Annual Rainfall Prediction System")
st.write(
    "Predict annual rainfall (in mm) based on meteorological indicators "
    "and historical precipitation records."
)

if pipeline is None:
    st.error(
        "❌ Trained model file (`rainfall_pred.pkl`) not found in the project root! "
        "Please train and export the model using `joblib.dump(model_pipeline, 'rainfall_pred.pkl')` first."
    )
    st.stop()

# --- Sidebar Configuration & Inputs ---
st.sidebar.header("📍 Location & Time")

indian_states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

selected_state = st.sidebar.selectbox("Select State", sorted(indian_states), index=sorted(indian_states).index("Uttar Pradesh") if "Uttar Pradesh" in indian_states else 0)
selected_year = st.sidebar.number_input("Forecast Year", min_value=1950, max_value=2050, value=2024, step=1)

# --- Main Form Input Section ---
st.subheader("🌤️ Meteorological Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    avg_temp = st.slider(
        "Average Temperature (°C)",
        min_value=5.0,
        max_value=45.0,
        value=24.5,
        step=0.1,
        help="Mean annual temperature in degrees Celsius"
    )
    avg_humidity = st.slider(
        "Average Humidity (%)",
        min_value=10.0,
        max_value=100.0,
        value=72.0,
        step=0.5,
        help="Mean relative atmospheric humidity percentage"
    )

with col2:
    avg_pressure = st.number_input(
        "Atmospheric Pressure (hPa)",
        min_value=950.0,
        max_value=1050.0,
        value=1010.5,
        step=0.1,
        help="Mean atmospheric surface pressure in hectopascals"
    )
    avg_wind_speed = st.slider(
        "Average Wind Speed (km/h)",
        min_value=0.0,
        max_value=50.0,
        value=8.5,
        step=0.1,
        help="Mean wind speed measured in km/h"
    )

with col3:
    avg_cloud_cover = st.slider(
        "Average Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
        help="Percentage of sky covered by clouds"
    )

st.subheader("📊 Historical Precipitation (Lag Features)")
col_hist1, col_hist2 = st.columns(2)

with col_hist1:
    prev_year_rainfall = st.number_input(
        "Previous Year Rainfall (mm)",
        min_value=0.0,
        max_value=6000.0,
        value=950.0,
        step=10.0,
        help="Total rainfall recorded in the immediate preceding year"
    )

with col_hist2:
    prev_3_year_avg = st.number_input(
        "Previous 3-Year Average Rainfall (mm)",
        min_value=0.0,
        max_value=6000.0,
        value=910.0,
        step=10.0,
        help="Rolling 3-year mean precipitation in millimeters"
    )

# --- Construct Input DataFrame ---
input_df = pd.DataFrame([{
    "Year": selected_year,
    "State": selected_state,
    "Avg_Temperature_C": float(avg_temp),
    "Avg_Humidity_percent": float(avg_humidity),
    "Avg_Pressure_hPa": float(avg_pressure),
    "Avg_Wind_Speed_kmh": float(avg_wind_speed),
    "Avg_Cloud_Cover_percent": float(avg_cloud_cover),
    "Previous_Year_Rainfall_mm": float(prev_year_rainfall),
    "Previous_3_Year_Avg_Rainfall_mm": float(prev_3_year_avg)
}])

st.markdown("---")

# --- Run Prediction ---
if st.button("🔮 Predict Annual Rainfall", use_container_width=True, type="primary"):
    try:
        prediction = pipeline.predict(input_df)[0]
        prediction_val = max(0.0, float(prediction))  # Rainfall cannot be negative

        st.success("✅ Prediction Generated Successfully!")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(
                label=f"Predicted Rainfall for {selected_state} ({selected_year})",
                value=f"{prediction_val:.2f} mm"
            )

        with res_col2:
            # Classification bucket guidance
            if prediction_val < 500:
                cat_label = "Low Rainfall / Arid Condition"
                cat_color = "warning"
            elif 500 <= prediction_val <= 1500:
                cat_label = "Moderate / Normal Rainfall"
                cat_color = "info"
            else:
                cat_label = "High / Heavy Precipitation"
                cat_color = "success"

            st.info(f"**Precipitation Category:** {cat_label}")
            
        with st.expander("🔍 View Processed Feature Payload"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.info(
            "Tip: Ensure that the pipeline was saved with a ColumnTransformer that includes "
            "both the scaler and OneHotEncoder."
        )