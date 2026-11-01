
import os
import streamlit as st
import pandas as pd
import joblib

#Load the model committed by the pipeline using correct double underscores
model_path = os.path.join(os.path.dirname(__file__),
              "best_tourism_package_prediction_model.joblib")

model = joblib.load(model_path)
st.title('Tourism package purchase prediction app')
st.write("""
The primary objective of this application is
predicts whether a customer will purchase the newly introduced Wellness
Tourism Package before contacting them or not.

Enter the TypeOfContact, CityTier, Occupation, Gender,
MaritalStatus, Passport, and MonthlyIncome data to get a prediction.
""")

TypeOfContact = st.selectbox('Select Contact Type',["Self Enquiry","Company Invited"])
CityTier = st.selectbox('Select City Tier',[1,2,3])
Occupation = st.selectbox('Select Occupation',["Salaried","Small Business","Large Business","Free Lancer"])
Gender = st.selectbox('Select Gender',["Male","Female"])
MaritalStatus = st.selectbox('Select Marital Status',["Single","Married","Divorced","Unmarried"])
Passport = st.selectbox('Select Passport',[1,0])
MonthlyIncome = st.number_input('Enter Monthly Income')

input_data = pd.DataFrame([{
    'TypeOfContact': TypeOfContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Passport': Passport,
    'MonthlyIncome': MonthlyIncome,
}])

if st.button('Predict Purchase'):
    prediction = model.predict(input_data)[0]
    result = "Package Purchased" if prediction == 1 else "Package Not Purchased"
    st.subheader('Prediction Result:')
    st.success(f"The model predicts: {result}")
