import streamlit as st
import pandas as pd
import joblib

# Load the Trained Model
model = joblib.load('model.pkl') 

def run():
    # Define All Expected Columns
    expected_columns = [
        'Employee ID', 'Age', 'Gender', 'Years at Company', 'Job Role', 'Monthly Income',
        'Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Number of Promotions',
        'Overtime', 'Distance from Home', 'Education Level', 'Marital Status', 'Number of Dependents',
        'Job Level', 'Company Size', 'Company Tenure', 'Remote Work', 'Leadership Opportunities',
        'Innovation Opportunities', 'Company Reputation', 'Employee Recognition', 'Attrition'
    ]

    # App Title
    st.title("Employee Attrition Prediction")
    st.markdown("Enter the details of an employee to predict whether they are likely to resign.")

    # Collect User Inputs
    st.sidebar.header("Employee Details")
    age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    job_role = st.sidebar.selectbox("Job Role", ["Sales", "Engineer", "Manager", "HR"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education_level = st.sidebar.slider("Education Level (1 = Low, 5 = High)", 1, 5, value=3)
    monthly_income = st.sidebar.number_input("Monthly Income (in $)", min_value=1000, max_value=20000, value=4000, step=500)
    years_at_company = st.sidebar.number_input("Years at Company", min_value=0, max_value=40, value=5, step=1)
    work_life_balance = st.sidebar.slider("Work-Life Balance (1 = Low, 5 = High)", 1, 5, value=3)
    job_satisfaction = st.sidebar.slider("Job Satisfaction (1 = Low, 5 = High)", 1, 5, value=3)
    performance_rating = st.sidebar.slider("Performance Rating (1 = Low, 5 = High)", 1, 5, value=3)
    overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
    distance_from_home = st.sidebar.number_input("Distance from Home (in km)", min_value=0, max_value=100, value=10, step=1)
    number_of_promotions = st.sidebar.number_input("Number of Promotions", min_value=0, max_value=10, value=1, step=1)
    number_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
    company_size = st.sidebar.selectbox("Company Size", ["Small", "Medium", "Large"])
    company_tenure = st.sidebar.number_input("Company Tenure (in years)", min_value=0, max_value=40, value=5, step=1)
    remote_work = st.sidebar.selectbox("Remote Work", ["Yes", "No"])
    leadership_opportunities = st.sidebar.selectbox("Leadership Opportunities", ["Yes", "No"])
    innovation_opportunities = st.sidebar.selectbox("Innovation Opportunities", ["Yes", "No"])
    company_reputation = st.sidebar.selectbox("Company Reputation", ["Poor", "Average", "Excellent"])
    employee_recognition = st.sidebar.selectbox("Employee Recognition", ["Yes", "No"])

    # Fill Missing Columns with Defaults
    default_values = {
        'Employee ID': 0,
        'Attrition': 0 
    }

    # Create Input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Job Role': [job_role],
        'Marital Status': [marital_status],
        'Education Level': [education_level],
        'Monthly Income': [monthly_income],
        'Years at Company': [years_at_company],
        'Work-Life Balance': [work_life_balance],
        'Job Satisfaction': [job_satisfaction],
        'Performance Rating': [performance_rating],
        'Overtime': [overtime],
        'Distance from Home': [distance_from_home],
        'Number of Promotions': [number_of_promotions],
        'Number of Dependents': [number_of_dependents],
        'Job Level': [job_level],
        'Company Size': [company_size],
        'Company Tenure': [company_tenure],
        'Remote Work': [remote_work],
        'Leadership Opportunities': [leadership_opportunities],
        'Innovation Opportunities': [innovation_opportunities],
        'Company Reputation': [company_reputation],
        'Employee Recognition': [employee_recognition]
    })

    # Add Missing Columns with Default Values
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = default_values.get(col, 0)

    # Display Input Data
    st.write("### Employee Details Entered")
    st.dataframe(input_data)

    # Make Predictions
    if st.button("Predict Attrition"):
        prediction = model.predict(input_data.drop(columns=['Attrition']))
        prediction_proba = model.predict_proba(input_data.drop(columns=['Attrition']))

        # Display Results
        if prediction[0] == 1:
            st.error(f"The employee is likely to resign with a probability of {prediction_proba[0][1] * 100:.2f}%.")
        else:
            st.success(f"The employee is likely to stay with a probability of {prediction_proba[0][0] * 100:.2f}%.")
