import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page config
st.set_page_config(page_title="Credit Card Attrition Prediction", layout="wide")

# Title
st.title("Credit Card Customer Attrition Prediction")
st.write("by Prasetyo Sukma Raharjo - DS32B")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "EDA", "Model Results", "Business Insights"])

# Project Overview
if page == "Project Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Description")
        st.write("Predicting credit card customers who are likely to churn (attrition)")
        
        st.subheader("Goals")
        st.write("""
        - Identify key factors causing customer churn
        - Build predictive model to classify at-risk customers
        - Provide business recommendations for retention
        """)
    
    with col2:
        st.subheader("Methods & Analysis")
        st.write("""
        - EDA: Statistical analysis, distributions, correlations
        - Data Preprocessing: Handling imbalance, scaling/encoding
        - ML Modeling: Model experimentation and evaluation
        """)
        
        st.subheader("Results")
        st.write("""
        - Key features influencing attrition
        - Best model with high accuracy and recall
        - Actionable business insights
        """)
    
    st.subheader("Dataset Information")
    st.write("""
    - 10,127 customers with 23 features
    - Approximately 16% churn rate
    - Mixed data types (categorical and numerical)
    """)

# EDA Section
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gender", "Marital Status", "Education", "Income", "Card Category"
    ])
    
    with tab1:
        st.subheader("Proportion by Gender")
        st.write("**Women have 3% higher churn rate than men**")
        st.write("p-value: 0.0001 (significant relationship)")
        
        # Gender plot data
        gender_data = pd.DataFrame({
            'Gender': ['Female', 'Male'],
            'Attrited': [17.4, 14.6],
            'Existing': [82.6, 85.4]
        })
        
        fig, ax = plt.subplots()
        gender_data.plot(x='Gender', y=['Attrited', 'Existing'], kind='bar', ax=ax)
        ax.set_ylabel('Proportion (%)')
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Proportion by Marital Status")
        st.write("**Married customers tend to stay**")
        st.write("p-value: 0.108 (not significant)")
        
        # Marital status data
        marital_data = pd.DataFrame({
            'Status': ['Married', 'Single', 'Unknown'],
            'Attrited': [15.1, 17.2, 12.6],
            'Existing': [84.9, 82.8, 87.4]
        })
        
        fig, ax = plt.subplots()
        marital_data.plot(x='Status', y=['Attrited', 'Existing'], kind='bar', ax=ax)
        ax.set_ylabel('Proportion (%)')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Proportion by Education Level")
        st.write("**Doctorate level tends to churn more (21%)**")
        st.write("p-value: 0.052 (not significant)")
        
        # Education data
        edu_data = pd.DataFrame({
            'Education': ['High School', 'Graduate', 'Post-Graduate', 'Uneducated', 'Unknown'],
            'Attrited': [15.2, 15.6, 21.1, 18.4, 8.3],
            'Existing': [84.8, 84.4, 78.9, 81.6, 91.7]
        })
        
        fig, ax = plt.subplots()
        edu_data.plot(x='Education', y=['Attrited', 'Existing'], kind='bar', ax=ax)
        ax.set_ylabel('Proportion (%)')
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Proportion by Income Category")
        st.write("**High income ($120K) has higher churn risk (17.3%)**")
        st.write("p-value: 0.025 (significant)")
        
        # Income data
        income_data = pd.DataFrame({
            'Income': ['<$40K', '$40K-60K', '$60K-80K', '$80K-120K', '>$120K'],
            'Attrited': [16.9, 15.2, 17.8, 15.6, 17.3],
            'Existing': [83.1, 84.8, 82.2, 84.4, 82.7]
        })
        
        fig, ax = plt.subplots()
        income_data.plot(x='Income', y=['Attrited', 'Existing'], kind='bar', ax=ax)
        ax.set_ylabel('Proportion (%)')
        st.pyplot(fig)
    
    with tab5:
        st.subheader("Proportion by Card Category")
        st.write("**Blue card dominates (>90% of customers)**")
        st.write("p-value: 0.52 (not significant)")
        
        # Card category data
        card_data = pd.DataFrame({
            'Card': ['Blue', 'Silver', 'Gold', 'Platinum'],
            'Attrited': [16.1, 15.2, 14.8, 12.3],
            'Existing': [83.9, 84.8, 85.2, 87.7]
        })
        
        fig, ax = plt.subplots()
        card_data.plot(x='Card', y=['Attrited', 'Existing'], kind='bar', ax=ax)
        ax.set_ylabel('Proportion (%)')
        st.pyplot(fig)
    
    st.subheader("Key Features for Modeling")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Features:**")
        st.write("""
        - Dependent_count
        - Total_Relationship_Count
        - Months_Inactive_12_mon
        - Contacts_Count_12_mon
        - Total_Trans_Ct
        - Avg_Utilization_Ratio
        - Revolving_Balance
        """)
    
    with col2:
        st.write("**Categorical Features:**")
        st.write("""
        - Gender
        - Income_category
        """)

# Model Results
elif page == "Model Results":
    st.header("Machine Learning Model Results")
    
    st.subheader("Model Comparison")
    
    # Model performance data
    model_data = pd.DataFrame({
        'Model': ['Naive Bayes', 'Logistic Regression', 'KNN (k=19)', 'XGBoost (SMOTE)'],
        'Accuracy': [0.8772, 0.8859, 0.8997, 0.9143],
        'Precision': [0.7069, 0.7185, 0.8097, 0.7474],
        'Recall': [0.4029, 0.4767, 0.4914, 0.7052],
        'F1-score': [0.5133, 0.5731, 0.6116, 0.7257],
        'AUC': [0.88, 0.90, 0.92, 0.95]
    })
    
    st.dataframe(model_data.style.highlight_max(axis=0))
    
    st.write("""
    **XGBoost with SMOTE** performed best for predicting customer attrition, 
    with the highest recall (70.5%) and F1-score (72.6%).
    """)
    
    st.subheader("Feature Importance")
    
    # Feature importance data
    features = [
        'Total_Trans_Ct_std', 'Avg_Utilization_Ratio_std',
        'Total_Relationship_Count_std', 'Months_inactive_12_mon_std',
        'Contacts_Count_12_mon_std', 'Dependent_count_std',
        'Income_Category_std', 'Gender_M'
    ]
    importance = [0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01, 0.01]
    
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax)
    ax.set_title('Feature Importance (XGBoost)')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)
    
    st.write("""
    **Key Findings:**
    - Total transaction count is the most important predictor
    - Utilization ratio and relationship count are also significant
    - Demographic features have lower importance
    """)

# Business Insights
elif page == "Business Insights":
    st.header("Business Insights & Recommendations")
    
    st.subheader("Boosting Transactions")
    st.write("""
    - Offer 5% cashback for minimum 3 transactions/month
    - Provide transaction tips via email or social media
    - Offer relevant marketplace discounts
    """)
    
    st.subheader("Strengthening Customer Relationships")
    st.write("""
    - Offer high-interest savings products for credit-only customers
    - Automatic upgrade to Silver/Gold cards for Blue cardholders with >Rp10M monthly transactions
    """)
    
    st.subheader("Improving Customer Service")
    st.write("""
    - Retrain CS team for faster resolution (24-hour SLA)
    - Review CRM data and conduct evaluations
    - Proactive outreach to high-risk customers
    """)
    
    st.subheader("Target Groups for Retention")
    st.write("""
    - **High-risk segments:**
        - Female customers
        - High-income ($120K+) customers
        - Customers with 3 credit cards
        - Doctorate-educated customers
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Contact:")
st.sidebar.write("Prasetyo Sukma Raharjo")
st.sidebar.write("prasetyo.sukmaraharjo@gmail.com")
st.sidebar.write("LinkedIn.com/in/prasetyosukmaraharjo")
