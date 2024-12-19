import streamlit as st
# Set Page Configuration
st.set_page_config(
    page_title="Employee Attrition App",
    page_icon="💼",
    layout="wide"
)

import matplotlib.pyplot as plt
from PIL import Image

# Load a Header Image
image = Image.open("image.jpg")
st.image(image, caption="Welcome to the Employee Attrition Checker", use_container_width=True)

# App Title and Intro
st.title("Employee Attrition Checker")
st.markdown("""
Welcome to the **Employee Attrition Checker Application**!  
This tool allows you to:
- **Explore the Dataset**: Analyze patterns and insights into employee attrition.
- **Predict Attrition**: Use a machine learning model to assess the risk of an employee resigning.  
Navigate through the options in the sidebar to get started!
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.selectbox(
    "Choose a Section:",
    ["Home", "EDA", "Prediction"]
)

import eda
import prediction

# Main Content Based on User Choice
if choice == "Home":
    st.header("🏠 Home")
    st.markdown("""
    Use the sidebar to navigate between:
    - **EDA**: For exploratory data analysis.
    - **Prediction**: To input details and predict attrition risk.
    """)

elif choice == "EDA":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("Uncover patterns and insights from the dataset.")
    eda.run()

elif choice == "Prediction":
    st.header("🔮 Prediction")
    st.markdown("Input employee details to predict whether they are likely to resign.")
    prediction.run()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by Michael Wilbert Puradisastra")
