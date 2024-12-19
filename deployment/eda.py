import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
        }
        [data-testid="stSidebar"] {
            background-color: #2e86de;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and Header
    st.markdown("<h1 style='text-align: center; color: #2e86de;'>Employee Attrition Checker</h1>", unsafe_allow_html=True)
    st.markdown("Welcome to the **EDA Section** of the Employee Attrition Checker Application. Gain insights into the dataset before building predictive models.")

    # Add Image
    image = Image.open("resign.jpg")
    st.image(image, caption='Resigning Worker', use_container_width=True)

    # Footer
    st.write("This page was created by **Michael Wilbert Puradisastra**")

    # Load Dataset
    df = pd.read_csv('train.csv')
    st.write("### Dataset Preview")
    st.dataframe(df)
    st.write("The dataset contains details about employees, including their job roles, salaries, and factors contributing to attrition.")

    # Dataset Information
    st.subheader("📊 Dataset Information")
    st.markdown("---")
    st.write(f"**Number of Rows and Columns:** {df.shape}")
    st.write("**Column Data Types and Non-Null Counts:**")
    buffer = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
    buffer.columns = ['Column Name', 'Data Type']
    buffer['Non-Null Count'] = df.notnull().sum().values
    st.dataframe(buffer)
    st.write("**Descriptive Statistics:**")
    st.write(df.describe())
    st.write("**Missing Values:**")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0] if missing_values.any() else "No missing values detected.")

    # Distribution of Target Variable
    st.subheader("Distribution of Target Variable (Attrition)")
    attrition_counts = df['Attrition'].value_counts()
    fig1 = px.pie(attrition_counts, names=attrition_counts.index, values=attrition_counts.values, title="Attrition Distribution")
    st.plotly_chart(fig1)

    # Numerical Feature Analysis
    st.subheader("Numerical Feature Analysis")
    st.write("### Distribution of Employee Age")
    fig2 = px.histogram(df, x='Age', title="Age Distribution")
    st.plotly_chart(fig2)

    # Categorical Feature Analysis
    st.subheader("📈 Categorical Feature Analysis")
    st.write("### Analyze Categorical Features")

    # Selectbox for categorical feature selection
    categorical_features = ['Job Role', 'Marital Status', 'Gender', 'Education Level', 'Overtime']
    selected_feature = st.selectbox("Select a categorical feature to analyze:", categorical_features)

    # Prepare the data for plotting
    category_counts = df[selected_feature].value_counts().reset_index()
    category_counts.columns = [selected_feature, 'Count']

    # Create the bar chart
    fig3 = px.bar(
        category_counts,
        x=selected_feature,
        y='Count',
        title=f"Distribution of {selected_feature}",
        labels={selected_feature: selected_feature, 'Count': 'Count'}
    )
    st.plotly_chart(fig3)

    st.write("### Distribution of Marital Status")

    # Prepare the data for plotting
    marital_status_counts = df['Marital Status'].value_counts().reset_index()
    marital_status_counts.columns = ['Marital Status', 'Count'] 

    # Create the bar chart
    fig4 = px.bar(
        marital_status_counts,
        x='Marital Status',
        y='Count',
        title="Marital Status Distribution",
        labels={'Marital Status': 'Marital Status', 'Count': 'Count'}
    )
    st.plotly_chart(fig4)

    # Relationship Between Features and Attrition
    st.subheader("Relationship Between Features and Attrition")
    st.write("### Monthly Income vs. Attrition")

    # Create the box plot
    fig5 = px.box(
        df,
        x='Attrition',
        y='Monthly Income',
        title="Monthly Income by Attrition",
        color='Attrition',
        labels={'Attrition': 'Attrition', 'Monthly Income': 'Monthly Income'}
    )
    st.plotly_chart(fig5)

    st.write("### Age vs. Attrition")
    fig6 = px.histogram(df, x='Age', color='Attrition', barmode='group', title="Age Distribution by Attrition")
    st.plotly_chart(fig6)

    # Multivariate Analysis
    st.subheader("Multivariate Analysis")
    st.write("### Pairplot of Selected Features")
    pairplot_features = ['Age', 'Monthly Income', 'Years at Company', 'Attrition']
    sns.pairplot(df[pairplot_features], hue='Attrition', diag_kind='kde')
    plt.savefig("pairplot.png")
    st.image("pairplot.png", caption="Pairplot of Age, Monthly Income, Years at Company, and Attrition", use_container_width=True)

    # Key Insights
    st.subheader("Key Insights")
    st.markdown("""
    - Most employees who resigned tend to have lower monthly income compared to those who stayed.
    - Younger employees with less experience are more likely to resign.
    - Certain job roles, such as Sales, have a higher rate of attrition compared to others.
    """)

    # Interactive Visualization
    st.subheader("🎨 Interactive Visualization")
    st.write("### Customize Scatter Plot")

    # Selectbox for choosing features for scatter plot
    numerical_features = ['Age', 'Monthly Income', 'Years at Company', 'Distance from Home']
    x_axis = st.selectbox("Select the X-axis feature:", numerical_features)
    y_axis = st.selectbox("Select the Y-axis feature:", numerical_features)

    # Create scatter plot
    fig7 = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="Attrition",
        title=f"{x_axis} vs {y_axis} Colored by Attrition",
        labels={x_axis: x_axis, y_axis: y_axis, 'Attrition': 'Attrition'}
    )
    st.plotly_chart(fig7)