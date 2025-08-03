import streamlit as st
import pandas as pd
from joblib import load


@st.cache_resource
def load_model():
    pipeline = load('powerful_fire_prediction_pipeline.joblib')
    return pipeline


st.title('Forest Fire Prevention')
st.subheader('Predict the probability of Forest-Fire Occurrence')

pipeline = load_model()

col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input('Temperature', min_value=10, max_value=40, value=25)
    isi = st.number_input('ISI', min_value=0.0, max_value=36.0, value=9.0)
    ws = st.number_input('Wind', min_value=2, max_value=30, value=15)

with col2:
    rh = st.number_input('RH', min_value=10, max_value=100, value=60)
    dmc = st.number_input('DMC', min_value=2.0, max_value=70.0, value=25.0)
    rain = st.number_input('Rain', min_value=0.0, max_value=20.0, value=0.0)

with col3:
    ffmc = st.number_input('FFMC', min_value=18.7, max_value=100.0, value=85.0)
    dc = st.number_input('DC', min_value=15.0, max_value=300.0, value=90.0)

region_name = st.sidebar.selectbox('Region', ('Bejaia', 'Sidi-Bel Abbes'))
region = 0 if region_name == 'Bejaia' else 1

if st.button('PREDICT PROBABILITY'):
    bui = dmc + 0.2 * (dc - dmc)
    fwi = 0.25 * (isi + bui)

    input_data = {
        'Temperature': [temp], 'RH': [rh], 'Ws': [ws], 'Rain': [rain],
        'FFMC': [ffmc], 'DMC': [dmc], 'DC': [dc], 'ISI': [isi],
        'BUI': [bui], 'FWI': [fwi], 'Region': [region]
    }
    input_df = pd.DataFrame(input_data)

    prediction_proba = pipeline.predict_proba(input_df)
    fire_probability = prediction_proba[0][1]

    st.subheader('Prediction Result')
    if fire_probability > 0.5:
        prob_text = f"Probability: {fire_probability*100:.2f}%"
        st.error(f'🔥 High Chance of Fire ({prob_text})')
    else:
        prob_text = f"Probability: {(1-fire_probability)*100:.2f}%"
        st.success(f'💧 Low Chance of Fire ({prob_text})')