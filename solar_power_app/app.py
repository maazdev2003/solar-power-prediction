import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import requests
import time

# 🛠️ Must be the first Streamlit command
st.set_page_config(page_title="Solar Power Prediction", page_icon="🌞", layout="centered")

# 🎨 Background Styling
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

# 🧠 Load model and scaler
model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# 🏷️ Title
st.title("🌞 Solar Power Generation Prediction")
st.markdown("### Predict solar power output using environmental or live weather data.")

# 🔘 Choose Input Mode
option = st.radio("Choose Input Mode:", ["Manual Input", "Live Weather (via City Name)"])

# ✨ Function to display prediction
def show_prediction(prediction):
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

# 📊 Gauge Chart Function
def show_gauge(prediction):
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
                {'range': [12000, 20000], 'color': "#f4cccc"}]
        }))
    st.plotly_chart(fig, use_container_width=True)

# ⚡ Power Table Function
def show_power_table():
    st.markdown("""
    <style>
    .power-table {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
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
        text-align: center;
    }
    .power-table th {
        color: #ffd700;
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
    """, unsafe_allow_html=True)

# 🌤️ Live Weather Mode
if option == "Live Weather (via City Name)":
    city = st.text_input("Enter City Name (e.g., Bangalore, Chennai, Mumbai)")
    API_KEY = "353f309ca8ee77fcba627bfc99c7f234"

    if st.button("Fetch Weather & Predict"):
        if city:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
                response = requests.get(url)
                data = response.json()

                if data["cod"] == 200:
                    st.subheader(f"🌤️ Live Weather for {city.capitalize()}")
                    temperature = data["main"]["temp"]
                    humidity = data["main"]["humidity"]
                    pressure = round(data["main"]["pressure"] * 0.02953, 2)
                    wind_speed = data["wind"]["speed"]
                    visibility = round(data.get("visibility", 10000) / 1000, 2)

                    st.write(f"**Temperature:** {temperature}°C")
                    st.write(f"**Humidity:** {humidity}%")
                    st.write(f"**Pressure:** {pressure} inHg")
                    st.write(f"**Wind Speed:** {wind_speed} m/s")
                    st.write(f"**Visibility:** {visibility} km")

                    # Default fixed inputs
                    distance_to_solar_noon = 0.5
                    wind_direction = 180
                    sky_cover = 2
                    avg_wind_speed = wind_speed
                    avg_pressure = pressure

                    input_data = np.array([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                            sky_cover, visibility, humidity, avg_wind_speed, avg_pressure]])
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]

                    show_prediction(prediction)
                    show_gauge(prediction)
                    show_power_table()
                else:
                    st.error("❌ City not found! Please check the spelling and try again.")
            except Exception:
                st.error("⚠️ Unable to fetch weather data. Please check your connection or API key.")
        else:
            st.warning("Please enter a valid city name.")

# 🧩 Manual Input Mode
else:
    distance_to_solar_noon = st.number_input("Distance to Solar Noon (radians)", 0.0, 2.0, 0.5)
    temperature = st.number_input("Temperature (°C)", 0, 100, 58)
    wind_direction = st.number_input("Wind Direction (degrees)", 0, 360, 25)
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 10.0)
    sky_cover = st.number_input("Sky Cover (0-4)", 0, 4, 2)
    visibility = st.number_input("Visibility (km)", 0.0, 20.0, 10.0)
    humidity = st.number_input("Humidity (%)", 0, 100, 70)
    avg_wind_speed = st.number_input("Average Wind Speed (3-hour period, m/s)", 0.0, 50.0, 10.0)
    avg_pressure = st.number_input("Average Pressure (inHg)", 0.0, 40.0, 30.0)

    if st.button("Predict Power Generation"):
        with st.spinner("🔍 Analyzing environmental data..."):
            time.sleep(2)
            input_data = np.array([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                    sky_cover, visibility, humidity, avg_wind_speed, avg_pressure]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

        show_prediction(prediction)
        show_gauge(prediction)
        show_power_table()

# ✨ Footer
st.write("---")
st.caption("Developed by Maaz Ahmed Risaldar | Solar Power Prediction Project")
