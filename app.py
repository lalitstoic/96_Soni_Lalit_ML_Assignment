import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("Heart Disease Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Patient Details:")

# Collect all 15 features
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol Level (chol)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise-Induced Angina (1 = Yes; 0 = No)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (0 = Normal; 1 = Fixed Defect; 2 = Reversible Defect)", [0, 1, 2])

# Convert categorical values
sex = 1 if sex == "Male" else 0

# Create the feature array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Make prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Patient has Heart Disease" if prediction == 1 else "No Heart Disease"
    
    st.subheader("Prediction Result:")
    st.write(result)
