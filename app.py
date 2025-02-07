import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = "./heart_disease_model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter your health parameters to check the risk of heart disease.")

# User input fields with default values
age = st.number_input("Enter age", min_value=0, max_value=120, step=1, value=50)
sex = st.selectbox("Select sex", ["Male", "Female"], index=0)
cp = st.selectbox("Select chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"], index=1)
trestbps = st.number_input("Enter resting blood pressure (trestbps)", min_value=0.0, step=1.0, value=130.0)
chol = st.number_input("Enter serum cholesterol (chol)", min_value=0.0, step=1.0, value=200.0)
fbs = st.selectbox("Is fasting blood sugar > 120 mg/dl? (fbs)", [True, False], index=1)
restecg = st.selectbox("Select resting electrocardiographic results (restecg)", ["normal", "lv hypertrophy"], index=0)
thalch = st.number_input("Enter maximum heart rate achieved (thalch)", min_value=0.0, step=1.0, value=150.0)
exang = st.selectbox("Does the patient have exercise-induced angina? (exang)", [True, False], index=1)
oldpeak = st.number_input("Enter ST depression induced by exercise relative to rest (oldpeak)", min_value=0.0, step=0.1, value=1.5)
slope = st.selectbox("Select the slope of the peak exercise ST segment (slope)", ["upsloping", "flat", "downsloping"], index=1)
ca = st.number_input("Enter number of major vessels colored by fluoroscopy (ca)", min_value=0.0, step=1.0, value=0.0)
thal = st.selectbox("Select thalassemia type (thal)", ["normal", "fixed defect", "reversable defect"], index=0)

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalch': [thalch],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Display the user input
st.subheader("User Input Parameters")
st.write(input_data)

# Predict button
if st.button("Predict"):
    # Convert categorical variables to numerical using one-hot encoding
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    
    # Ensure the input data has the same columns as the model expects
    # Add missing columns with 0s
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in input_data_encoded.columns:
            input_data_encoded[feature] = 0
    
    # Reorder columns to match the model's expected input
    input_data_encoded = input_data_encoded[model_features]
    
    # Make prediction
    prediction = model.predict(input_data_encoded)
    result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
    
    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(result)