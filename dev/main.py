import streamlit as st
import requests
import pandas as pd

# Define the FastAPI endpoint URLs
API_URLS = {
    'LGBM': 'http://127.0.0.1:8000/predict_sepsis/lgbm',
    'RandomForest': 'http://127.0.0.1:8000/predict_sepsis/random_forest',
    'SVM': 'http://127.0.0.1:8000/predict_sepsis/svm'
}

# Streamlit app layout
st.title('Sepsis Prediction App')
st.write('Enter patient data to get sepsis predictions from different models.')

# Input fields for the patient data
PRG = st.number_input('PRG', min_value=0)
PL = st.number_input('PL', min_value=0)
PR = st.number_input('PR', min_value=0)
SK = st.number_input('SK', min_value=0)
TS = st.number_input('TS', min_value=0)
BD2 = st.number_input('BD2', format='%.2f')
Age = st.number_input('Age', min_value=0)

# Prepare the data to send to FastAPI
data = {
    'PRG': PRG,
    'PL': PL,
    'PR': PR,
    'SK': SK,
    'TS': TS,
    'BD2': BD2,
    'Age': Age
}

# Buttons to request predictions from different models
if st.button('Predict using LGBM'):
    response = requests.post(API_URLS['LGBM'], json=data)
    result = response.json()
    st.write(result)

if st.button('Predict using Random Forest'):
    response = requests.post(API_URLS['RandomForest'], json=data)
    result = response.json()
    st.write(result)

if st.button('Predict using SVM'):
    response = requests.post(API_URLS['SVM'], json=data)
    result = response.json()
    st.write(result)
