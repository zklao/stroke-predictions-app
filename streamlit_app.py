import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random


def get_data():
    model, scaler = pickle.load(open('stroke_mdl.pkl', 'rb'))
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return model, scaler, df


def yesNo(n):
    if n == 'Yes':
        return 1
    return 0


model, scaler, df = get_data()

st.title("Stroke Predictions App")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

age = st.slider('Age', 0, 100, 50)
gender = st.radio(label='Gender', options=['Female', 'Male', 'Other'])
married = st.radio(label='Ever married', options=['Yes', 'No'])
hypertension = st.radio(label='Hypertension', options=['Yes', 'No'])
heart_disease = st.radio(label='Heart disease', options=['Yes', 'No'])
glucose = st.slider('Average glucose level', 0, 300, value=int(df['avg_glucose_level'].mean()))
bmi = st.slider('BMI', 0, 70, value=int(df['bmi'].mean()))
smoked = st.selectbox('Smoking status', df['smoking_status'].unique())

if gender == 'Female':
    gender = 0
elif gender == 'Male':
    gender = 1
else:
    gender = random.choice([0, 1])

value_list = [1, 2, 3, 4]
smoking_dict = {key: value for key, value in zip(df['smoking_status'].unique(), value_list)}
print(smoking_dict)

user_values= [[gender, age, yesNo(hypertension), yesNo(heart_disease), yesNo(married), glucose, bmi, smoking_dict.get(smoked)]]

if st.button('Predict'):
    result = model.predict(scaler.transform(user_values))
    risk = 'The stroke risk is '
    if result == 1:
        risk += "HIGHT"
    else:
        risk += "LOW"
    
    st.subheader(risk)