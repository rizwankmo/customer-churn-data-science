#Gende --> 1 Female  0Male
#Churn --> 1Yes  0 No
#Scaler is exported as scaler.pkl
#Model is exported as model.pkl
#Order of the X -> 'Age' , 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and press predict button for predctions.")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter tenure", min_value=0, max_value=130, value=30)
monthlycharge = st.number_input("Enter Monthly CHarge", min_value=30, max_value=150)
gender = st.selectbox("enter gender",["Male","Female"])

st.divider()

predictbutton = st.button("PREDICT")

if predictbutton:
    
    gender_selected = 1 if gender == "Female" else 0 
    X= [age, gender_selected, tenure, monthlycharge]
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    predicted = "Yes" if prediction == 1 else "No"
    st.balloons()
    st.write(f"Predicted: {predicted}")
else:
    st.write("Please enter the values and press predict button")

