import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import time

# 🛠️ Must be the first Streamlit command
st.set_page_config(page_title="Solar Power Prediction", page_icon="🌞", layout="centered")

# 🎨 Background & Styling
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://www.nationalgrid.com/sites/default/files/images/EnergyExplained_DifferentTypesRenewableEnergy_640x360.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
background: rgba(0,0,0,0);
}

.block-container {
background: rgba(255, 255, 255, 0.15);
backdrop-filter: blur(15px);
-webkit-backdrop-filter: blur(15px);
padding: 2rem 3rem;
border-radius: 20px;
box-shadow: 0 0 25px rgba(255, 255, 255, 0.3);
border: 1px solid rgba(255, 255, 255, 0.4);
color: #000000;
}

h1 {
text-align: center;
color: #ffffff;
text-shadow: 2px 2px 10px #000000;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# 🧠 Safe model and scaler loading (works on Streamlit Cloud too)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_random_forest_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 🏷️ Title
st.title("🌞 Solar Power Generation Prediction")
st.markdown("### Predict solar power output based on environmental conditions.")

# 🧩 Input fields
distance_to_solar_noon = st.number_input("Distance to Solar Noon (radians)", 0.0, 2.0, 0.5)
temperature = st.number_input("Temperature (°C)", 0, 100, 58)
wind_direction = st.number_input("Wind Direction (degrees)", 0, 360, 25)
wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 10.0)
sky_cover = st.number_input("Sky Cover (0-4)", 0, 4, 2)
visibility = st.number_input("Visibility (km)", 0.0, 20.0, 10.0)
humidity = st.number_input("Humidity (%)", 0, 100, 70)
avg_wind_speed = st.number_input("Average Wind Speed (3-hour period, m/s)", 0.0, 50.0, 10.0)
avg_pressure = st.number_input("Average Pressure (inHg)", 0.0, 40.0, 30.0)

# ⚡ Prediction button
if st.button("Predict Power Generation"):
    with st.spinner('🔍 Analyzing environmental data and predicting power output...'):
        time.sleep(2)  # Simulate processing delay
        input_data = np.array([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                sky_cover, visibility, humidity, avg_wind_speed, avg_pressure]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

    # 🌟 Glowing animated text
    st.markdown(
        f"""
        <style>
        @keyframes glow {{
            0% {{ text-shadow: 0 0 5px #ff4b4b, 0 0 10px #ff4b4b, 0 0 20px #ff4b4b; }}
            50% {{ text-shadow: 0 0 20px #ff0000, 0 0 30px #ff4b4b, 0 0 40px #ff0000; }}
            100% {{ text-shadow: 0 0 5px #ff4b4b, 0 0 10px #ff4b4b, 0 0 20px #ff4b4b; }}
        }}
        .glow-text {{
            color: #ff0000;
            text-align: center;
            font-weight: bold;
            font-size: 28px;
            animation: glow 1.5s ease-in-out infinite alternate;
        }}
        </style>
        <h3 class="glow-text">🌞 Predicted Power Generated: {prediction:.2f} Joules (approx)</h3>
        """,
        unsafe_allow_html=True
    )

    # 📊 Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Predicted Power Output (Joules)"},
        gauge={
            'axis': {'range': [None, 20000]},
            'bar': {'color': "#fcbf49"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5000], 'color': "#d3f8e2"},
                {'range': [5000, 12000], 'color': "#ffe599"},
                {'range': [12000, 20000], 'color': "#f4cccc"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 📘 Power Level Table
    st.markdown("""
    <style>
    .power-table {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 12px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    color: #fff;
    font-weight: bold;
    margin-top: 20px;
    }
    .power-table table {
    width: 100%;
    border-collapse: collapse;
    }
    .power-table th, .power-table td {
    padding: 8px 12px;
    text-align: center;
    }
    .power-table th {
    color: #ffd700;
    }
    .note-footer {
    text-align: center;
    color: #fff;
    font-size: 14px;
    margin-top: 8px;
    font-style: italic;
    opacity: 0.9;
    }
    </style>

    <div class="power-table">
    <h4>⚡ Power Level Interpretation</h4>
    <table>
    <tr><th>Range (Joules)</th><th>Power Level</th><th>Meaning</th></tr>
    <tr><td>0 – 5000 J</td><td>🔴 Low</td><td>Cloudy / Poor sunlight</td></tr>
    <tr><td>5000 – 12000 J</td><td>🟡 Medium</td><td>Moderate sunlight</td></tr>
    <tr><td>12000 – 20000 J</td><td>🟢 High</td><td>Excellent solar conditions</td></tr>
    </table>
    </div>

    <div class="note-footer">
    Note: Prediction is based on current environmental conditions and model learning.
    </div>
    """, unsafe_allow_html=True)

# ✨ Footer
st.write("---")
st.caption("Solar Power Prediction Project")

