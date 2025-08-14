This project aims to predict credit card customers who are likely to churn (attrition) using machine learning. The model is built with Python and XGBoost, and deployed with Streamlit for interactive analysis.

Streamlit App
Visit the deployed Streamlit application:
https://finalprojectdscredit-card-attrition-prediction-il4n326jh4epfr7.streamlit.app/

Project Overview
Key Features
Attrition Prediction: Uses an XGBoost model with SMOTE to predict customer churn with 91.4% accuracy

Exploratory Data Analysis: Visualizations of key factors influencing attrition (gender, income, card category, etc.)

Model Comparison: Compares performance of Naive Bayes, Logistic Regression, KNN, and XGBoost

Feature Importance: Shows which factors most influence churn predictions

Business Recommendations: Actionable insights for customer retention strategies

Key Insights
Women have 3% higher churn rate than men (p=0.0001)

High-income ($120K+) customers have significantly higher churn risk (p=0.025)

Doctorate-educated customers show highest churn tendency (21%)

Total transaction count is the most important predictor of churn

Technologies Used
Python

Streamlit (for web app deployment)

Pandas & NumPy (data manipulation)

Matplotlib & Seaborn (visualizations)

XGBoost (machine learning model)

Imbalanced-learn (SMOTE for handling class imbalance)
