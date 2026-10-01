import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Mobile Price Range Predictor", page_icon="📱")

st.title("📱 Mobile Price Range Predictor")
st.write("Enter mobile specifications to predict its price range.")

MODEL_FILE = "best_mobile_price_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("Model file not found. First run 05_GridSearchCV_Random_Forest_Mobile_Price.ipynb or train_best_model.py.")
    st.stop()

model = joblib.load(MODEL_FILE)

st.subheader("Mobile Specifications")

c1, c2 = st.columns(2)

with c1:
    battery_power = st.number_input("Battery Power (mAh)", 500, 2000, 1200)
    blue = st.selectbox("Bluetooth", [0, 1], index=1)
    clock_speed = st.number_input("Clock Speed (GHz)", 0.1, 3.0, 1.5, step=0.1)
    dual_sim = st.selectbox("Dual SIM", [0, 1], index=1)
    fc = st.number_input("Front Camera (MP)", 0, 20, 5)
    four_g = st.selectbox("4G", [0, 1], index=1)
    int_memory = st.number_input("Internal Memory (GB)", 2, 64, 32)
    m_dep = st.number_input("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1)
    mobile_wt = st.number_input("Mobile Weight (g)", 80, 200, 140)

with c2:
    n_cores = st.number_input("Number of Cores", 1, 8, 4)
    pc = st.number_input("Primary Camera (MP)", 0, 21, 10)
    px_height = st.number_input("Pixel Resolution Height", 0, 2000, 800)
    px_width = st.number_input("Pixel Resolution Width", 0, 2000, 1200)
    ram = st.number_input("RAM (MB)", 256, 4000, 2000)
    sc_h = st.number_input("Screen Height (cm)", 5, 20, 12)
    sc_w = st.number_input("Screen Width (cm)", 2, 20, 7)
    talk_time = st.number_input("Talk Time (hours)", 2, 20, 10)
    three_g = st.selectbox("3G", [0, 1], index=1)
    touch_screen = st.selectbox("Touch Screen", [0, 1], index=1)
    wifi = st.selectbox("WiFi", [0, 1], index=1)

input_data = pd.DataFrame([{
    "battery_power": battery_power,
    "blue": blue,
    "clock_speed": clock_speed,
    "dual_sim": dual_sim,
    "fc": fc,
    "four_g": four_g,
    "int_memory": int_memory,
    "m_dep": m_dep,
    "mobile_wt": mobile_wt,
    "n_cores": n_cores,
    "pc": pc,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": sc_h,
    "sc_w": sc_w,
    "talk_time": talk_time,
    "three_g": three_g,
    "touch_screen": touch_screen,
    "wifi": wifi
}])

if st.button("Predict Price Range"):
    prediction = int(model.predict(input_data)[0])
    labels = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }
    st.success(f"Predicted Price Range: {prediction} - {labels[prediction]}")
