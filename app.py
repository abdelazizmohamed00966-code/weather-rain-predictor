
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


st.set_page_config(
    page_title="Weather Rain Predictor",
    page_icon="🌦️",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not os.path.exists('weather_model.pkl'):
        return None, None
    model   = joblib.load('weather_model.pkl')
    columns = joblib.load('feature_columns.pkl')
    return model, columns

model, feature_columns = load_model()


st.markdown("""
    <h1 style='text-align:center; color:#1565C0;'>🌦️ Weather Rain Predictor</h1>
    <p style='text-align:center; color:gray;'>
        Powered by Random Forest | Machine Learning Project
    </p>
""", unsafe_allow_html=True)

st.divider()

if model is None:
    st.error("❌ الموديل مش موجود! شغّل train_model.py الأول.")
    st.code("python train_model.py", language="bash")
    st.stop()


st.markdown("### 🌡️ أدخل بيانات الطقس:")


default_ranges = {
    'temperature':      (0.0,   50.0,  25.0,  0.1),
    'temp':             (0.0,   50.0,  25.0,  0.1),
    'mintemp':          (-10.0, 40.0,  12.0,  0.1),
    'maxtemp':          (0.0,   55.0,  30.0,  0.1),
    'humidity':         (0.0,   100.0, 60.0,  1.0),
    'humidity9am':      (0.0,   100.0, 55.0,  1.0),
    'humidity3pm':      (0.0,   100.0, 45.0,  1.0),
    'pressure':         (980.0, 1040.0,1013.0,0.1),
    'pressure9am':      (980.0, 1040.0,1015.0,0.1),
    'pressure3pm':      (980.0, 1040.0,1012.0,0.1),
    'windspeed':        (0.0,   130.0, 20.0,  0.5),
    'windspeed9am':     (0.0,   130.0, 15.0,  0.5),
    'windspeed3pm':     (0.0,   130.0, 25.0,  0.5),
    'rainfall':         (0.0,   200.0, 2.0,   0.1),
    'sunshine':         (0.0,   14.0,  7.0,   0.1),
    'evaporation':      (0.0,   50.0,  5.0,   0.1),
    'cloud9am':         (0.0,   9.0,   4.0,   1.0),
    'cloud3pm':         (0.0,   9.0,   4.0,   1.0),
    'dewpoint':         (-10.0, 35.0,  15.0,  0.1),
    'windgustspeed':    (0.0,   150.0, 40.0,  0.5),
    'temp9am':          (0.0,   45.0,  18.0,  0.1),
    'temp3pm':          (0.0,   50.0,  28.0,  0.1),
}

user_input = {}
cols = st.columns(2)

for i, col_name in enumerate(feature_columns):
    key = col_name.lower().replace(' ', '')
    if key in default_ranges:
        mn, mx, default, step = default_ranges[key]
    else:
        mn, mx, default, step = 0.0, 100.0, 50.0, 1.0

    with cols[i % 2]:
        user_input[col_name] = st.slider(
            label=col_name,
            min_value=float(mn),
            max_value=float(mx),
            value=float(default),
            step=float(step)
        )

st.markdown("")
if st.button("🔍  Predict Rain Tomorrow", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]

    st.divider()

    if prediction == 1:
        st.error(f"""
        ## 🌧️ Rain Expected Tomorrow!
        Confidence: **{proba[1]*100:.1f}%**
        """)
    else:
        st.success(f"""
        ## ☀️ No Rain Tomorrow
        Confidence: **{proba[0]*100:.1f}%**
        """)

    col1, col2 = st.columns(2)
    col1.metric("☀️ No Rain", f"{proba[0]*100:.1f}%")
    col2.metric("🌧️ Rain",    f"{proba[1]*100:.1f}%")

    st.progress(float(proba[1]), text=f"Rain probability: {proba[1]*100:.1f}%")


with st.sidebar:
    st.markdown("## 📊 Project Info")
    st.markdown("""
    | Item | Detail |
    |---|---|
    | **Algorithm** | Random Forest |
    | **Dataset** | Kaggle Weather |
    | **Task** | Rain Prediction |
    | **Train split** | 80% |
    | **Test split** | 20% |
    """)

    st.divider()
    st.markdown("## 👥 Team Members")
    st.markdown("""
    - Abdelaziz Mohamed Elkhadrgy : 82511010
    - Yossef Sherif Mostafa : 8251908
    - Abdelrahman Talaal Reda : 82511069
    - Wesam Fares Bashir : ‏8251944
    - Eyad Ziad El-Sayed : 8251544
    



    st.divider()
    st.markdown("## 📖 How it works")
    st.markdown("""
    1. User enters weather measurements
    2. Random Forest predicts rain
    3. Shows probability of rain/no rain
    """)

    if os.path.exists('feature_importance.png'):
        st.divider()
        st.markdown("## 📈 Feature Importance")
        st.image('feature_importance.png')

    st.caption("Machine Learning Project | Weather Forecast")
