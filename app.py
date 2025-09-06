import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_svm_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

st.title(" Iris Flower Prediction App")
st.write("Enter flower measurements and get the predicted species!")


# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    species = ["Setosa ", "Versicolor ", "Virginica "]
    st.success(f"Prediction: **{species[prediction]}**")