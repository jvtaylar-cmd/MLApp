 import streamlit as st
import joblib
import json

# Load the models
decision_tree_model = joblib.load("models/DecisionTree.pkl")
knn_model = joblib.load("models/KNN.pkl")
logistic_regression_model = joblib.load("models/logistic_regression_model.pkl")

# Create a Streamlit app title
st.title('Machine Learning Model Predictions')

# Select model
model_choice = st.selectbox('Select Model', ['Decision Tree', 'KNN', 'Logistic Regression'])

# Get input data
input_data = st.text_area('Enter input data as JSON')

# Make prediction
if st.button('Predict'):
    try:
        data = json.loads(input_data)  # Parse JSON input

        if model_choice == 'Decision Tree':
            model = decision_tree_model
        elif model_choice == 'KNN':
            model = knn_model
        else:
            model = logistic_regression_model

        prediction = model.predict([data])[0]
        st.write(f"Prediction: {prediction}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
