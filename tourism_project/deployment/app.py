import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the model and preprocessor
model = joblib.load('tourism_project/model_building/models/logistic_regression_model.joblib')

# Function to preprocess input data
def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    # The preprocessor is part of the loaded pipeline, so we just need to prepare the input DataFrame.
    # Ensure the columns are in the same order as during training if possible, though the pipeline handles feature names.
    return data

# Streamlit App
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="centered")

st.title("🌴 Wellness Tourism Package Predictor")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Input fields for customer details
st.sidebar.header("Customer Details")

customer_id = st.sidebar.text_input("CustomerID", "200000")
age = st.sidebar.slider("Age", 18, 90, 41)
type_of_contact = st.sidebar.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation = st.sidebar.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
number_of_person_visiting = st.sidebar.slider("Number of People Visiting", 1, 10, 2)
preferred_property_star = st.sidebar.slider("Preferred Property Star (1-5)", 1, 5, 3)
marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
number_of_trips = st.sidebar.slider("Number of Trips Annually", 1, 50, 5)
passport = st.sidebar.checkbox("Has Passport?")
own_car = st.sidebar.checkbox("Owns Car?")
number_of_children_visiting = st.sidebar.slider("Number of Children Visiting (<5 years)", 0, 5, 0)
designation = st.sidebar.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'VP', 'Director', 'Junior Executive', 'AVP', 'Assistant Manager'])
monthly_income = st.sidebar.number_input("Monthly Income", 0, 1000000, 25000)
pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
product_pitched = st.sidebar.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
number_of_followups = st.sidebar.slider("Number of Follow-ups", 0, 10, 3)
duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 1, 60, 10)

# Create a DataFrame from inputs
input_data = pd.DataFrame([{
    'CustomerID': int(customer_id),
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': 1 if passport else 0,
    'OwnCar': 1 if own_car else 0,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch,
}])

# Adding a dummy 'Unnamed: 0' column if the model expects it, based on original df structure
# This assumes 'Unnamed: 0' was just an index in the original CSV and not a meaningful feature
# If it was a meaningful feature, its input would need to be captured.
# This check might need to be more robust if 'Unnamed: 0' is truly not always present or is dynamic.
if 'Unnamed: 0' in model.named_steps['preprocessor'].get_feature_names_out(): # Using get_feature_names_out for safety
    if 'Unnamed: 0' not in input_data.columns:
        input_data.insert(0, 'Unnamed: 0', 0) # Placeholder, adjust if 'Unnamed: 0' has actual meaning

# Predict button
if st.sidebar.button("Predict Purchase"):
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The customer is likely to purchase the Wellness Tourism Package! (Probability: {prediction_proba:.2f})")
    else:
        st.info(f"The customer is not likely to purchase the Wellness Tourism Package. (Probability: {prediction_proba:.2f})")

    st.write("## Input Data:")
    st.dataframe(input_data)
