import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import plotly.express as px

from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f7f9fc;}
        .stTabs [data-baseweb="tab-list"] button {background-color: #d1e7ff; color: black; border-radius: 8px; margin: 5px;}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {background-color: #0d6efd; color: white;}
        .stMarkdown h3 {color: #0d6efd;}
    </style>
""", unsafe_allow_html=True)

# Navigation menu
selected = option_menu(menu_title=None, options=["📖 Business Case Presentation", "📊 Data Vizualization", "🤖 Prediction Models"], default_index=0, orientation="horizontal")

df = pd.read_csv("insurance.csv")

if selected == "📖 Business Case Presentation":
    st.title("🏥 Insurance Charges Analysis and Prediction")

    st.markdown("""
    ### 🔍 **Introduction**
    The insurance market is significantly influenced by various demographic and lifestyle factors. This dashboard explores how elements like **age**, **BMI**, **smoking status**, and **region** affect insurance charges, offering valuable insights for insurance companies to optimize pricing strategies and for individuals to understand potential premiums.
    """)

    st.markdown("""
    ### 📂 **Dataset Overview**
    - **Source:** Sample insurance dataset with demographic and health-related attributes.
    - **Size:** 1,338 records
    - **Features:** Age, Sex, BMI, Number of Children, Smoking Status, Region, and Insurance Charges.
    - **Modifications:**
    - No missing values
    - Clean and ready for analysis
    """)

    st.markdown("""
    ### 🎯 **Project Goals**
    1. **Exploratory Data Analysis (EDA):** Understand how different factors influence insurance charges.
    2. **Data Visualization:** Provide intuitive visual insights.
    3. **Predictive Modeling:** Estimate insurance charges based on user inputs.
    """)

    st.success("This project leverages regression modeling and interactive dashboards to uncover insurance charge trends and deliver on-the-fly charge estimations.")

    st.header("📊 Data Preview")
    st.dataframe(df.head())

    # Visualizing average charges by smoker status
    st.subheader("💸 Average Charges by Smoker Status")
    avg_charges_smoker = df.groupby('smoker')['charges'].mean().reset_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(data=avg_charges_smoker, x='smoker', y='charges', palette="viridis", ax=ax1)
    ax1.set_title("Average Insurance Charges by Smoker Status")
    st.pyplot(fig1)

    # Charges distribution by region
    st.subheader("🌎 Average Charges by Region")
    avg_charges_region = df.groupby('region')['charges'].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(data=avg_charges_region, x='region', y='charges', palette="pastel", ax=ax2)
    ax2.set_title("Average Charges by Region")
    st.pyplot(fig2)

    # Scatterplot of BMI vs Charges colored by smoking status
    st.subheader("📈 BMI vs. Insurance Charges by Smoking Status")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', palette="Set1", ax=ax3)
    ax3.set_title("BMI vs. Charges")
    st.pyplot(fig3)

elif selected == "🤖 Prediction Models":
    st.subheader("🧮 Estimate Insurance Charges")
    age_input = st.slider("Select Age", 18, 65, 30)
    bmi_input = st.slider("Select BMI", 15.0, 50.0, 25.0)
    children_input = st.slider("Number of Children", 0, 5, 0)
    smoker_input = st.selectbox("Smoker Status", ["yes", "no"])

    # Preparing data for model
    model_df = df.copy()
    model_df['smoker'] = model_df['smoker'].map({'yes': 1, 'no': 0})
    X = model_df[['age', 'bmi', 'children', 'smoker']]
    y = model_df['charges']

    model = LinearRegression()
    model.fit(X, y)

    smoker_val = 1 if smoker_input == 'yes' else 0
    predicted_charge = model.predict([[age_input, bmi_input, children_input, smoker_val]])[0]

    st.success(f"**Estimated Insurance Charge: ${predicted_charge:,.2f}**")
