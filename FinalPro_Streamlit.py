import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import shap

# Set page config
st.set_page_config(page_title="Customer Attrition Prediction", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("BankChurners.csv")
        # Clean column names (remove spaces and special characters)
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        # Drop unnecessary columns
        df = df.drop(['naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1',
                     'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2'], axis=1)
        return df
    except FileNotFoundError:
        st.error("File 'BankChurners.csv' not found. Please upload the dataset.")
        return None

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Model Building", "Business Insights"])

# Main content
if df is not None:
    if section == "Home":
        st.title("Credit Card Customer Attrition Prediction")
        st.markdown("""
        ### Overview Project
        This project aims to predict customer attrition for a credit card company using machine learning.
        
        **Data Characteristics:**
        - Categorical features: marital status, card category, income category, attrition flag
        - Numerical features: total transaction amount, transaction count, months on book, utilization ratio
        
        **Project Workflow:**
        1. Exploratory Data Analysis (EDA)
        2. Data Preprocessing
        3. Machine Learning Model Building
        4. Business Insights and Recommendations
        """)
        
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*_8jJ9yN4k6QYxQj0yQ7Z2g.png", width=600)
        
    elif section == "Data Overview":
        st.title("Dataset Overview")
        
        st.markdown("""
        ### Dataset Information
        - Number of customers: 10,127
        - Features: 23 columns
        - Attrition rate: ~16%
        """)
        
        st.subheader("Preview of the Dataset")
        st.dataframe(df.head())
        
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        st.subheader("Missing Values Check")
        st.write(df.isnull().sum())
        
    elif section == "EDA":
        st.title("Exploratory Data Analysis")
        
        # Attrition by Gender
        st.subheader("Proportion by Gender")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='gender', hue='attrition_flag', ax=ax)
        ax.set_title("Attrition by Gender")
        st.pyplot(fig)
        st.markdown("""
        **Insight:** 
        - There's a 3% higher attrition rate among women compared to men.
        - The p-value of 0.0001 indicates a significant relationship.
        """)
        
        # Attrition by Marital Status
        st.subheader("Proportion by Marital Status")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='marital_status', hue='attrition_flag', ax=ax)
        ax.set_title("Attrition by Marital Status")
        st.pyplot(fig)
        st.markdown("""
        **Insight:** 
        - Married customers are more likely to retain their cards compared to single customers.
        - The p-value of 0.108 indicates no significant relationship.
        """)
        
        # Attrition by Education Level
        st.subheader("Proportion by Education Level")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='education_level', hue='attrition_flag', ax=ax)
        ax.set_title("Attrition by Education Level")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("""
        **Insight:** 
        - Doctorate level has higher attrition tendency (21%) compared to other education levels.
        - The p-value of 0.052 indicates no significant relationship.
        """)
        
        # Attrition by Income Category
        st.subheader("Proportion by Income Category")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='income_category', hue='attrition_flag', ax=ax)
        ax.set_title("Attrition by Income Category")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("""
        **Insight:** 
        - The high income category ($120K) has a significant attrition risk (17.3%).
        - The p-value of 0.025 indicates a significant relationship.
        """)
        
        # Attrition by Card Category
        st.subheader("Proportion by Card Category")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='card_category', hue='attrition_flag', ax=ax)
        ax.set_title("Attrition by Card Category")
        st.pyplot(fig)
        st.markdown("""
        **Insight:** 
        - Blue card dominates the customer base (>90%).
        - The p-value of 0.52 indicates no significant relationship.
        """)
        
        # Numerical Features Analysis
        st.subheader("Numerical Features Analysis")
        
        # Correlation Matrix
        st.write("Correlation Matrix")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Boxplots for key features
        st.write("Boxplots for Key Features")
        features_to_plot = ['dependent_count', 'total_relationship_count', 'months_inactive_12_mon', 
                           'contacts_count_12_mon', 'total_trans_ct', 'avg_utilization_ratio']
        
        for feature in features_to_plot:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='attrition_flag', y=feature, ax=ax)
            ax.set_title(f"Distribution of {feature} by Attrition Status")
            st.pyplot(fig)
        
    elif section == "Model Building":
        st.title("Machine Learning Model Building")
        
        # Data Preprocessing
        st.subheader("Data Preprocessing")
        
        # Encode target variable
        df['attrition_flag'] = df['attrition_flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
        
        # Select features based on EDA
        categorical_features = ['gender', 'education_level', 'marital_status', 'income_category', 'card_category']
        numerical_features = ['customer_age', 'dependent_count', 'months_on_book', 
                             'total_relationship_count', 'months_inactive_12_mon', 
                             'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
                             'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
                             'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio']
        
        # Label encoding for categorical features
        le = LabelEncoder()
        for col in categorical_features:
            df[col] = le.fit_transform(df[col])
        
        # Feature selection
        X = df[categorical_features + numerical_features]
        y = df['attrition_flag']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        
        st.write("Preprocessing steps completed!")
        st.write(f"Original train shape: {X_train.shape}, After SMOTE: {X_train_smote.shape}")
        
        # Model Training and Evaluation
        st.subheader("Model Training and Evaluation")
        
        # Initialize models
        models = {
            "Naïve Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "XGBoost": XGBClassifier()
        }
        
        # Train and evaluate models
        results = []
        for name, model in models.items():
            # Train model
            if name == "XGBoost":
                model.fit(X_train_smote, y_train_smote)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Predictions
            if name == "XGBoost":
                train_pred = model.predict(X_train_smote)
                test_pred = model.predict(X_test_scaled)
                y_train_eval = y_train_smote
            else:
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                y_train_eval = y_train
            
            # Calculate metrics
            train_acc = accuracy_score(y_train_eval, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            train_f1 = f1_score(y_train_eval, train_pred)
            test_f1 = f1_score(y_test, test_pred)
            
            results.append({
                "Model": name,
                "Training Accuracy": f"{train_acc:.2%}",
                "Test Accuracy": f"{test_acc:.2%}",
                "Training F1-score": f"{train_f1:.2%}",
                "Test F1-score": f"{test_f1:.2%}"
            })
        
        # Display results
        st.table(pd.DataFrame(results))
        
        # Hyperparameter tuning for XGBoost
        st.subheader("XGBoost Hyperparameter Tuning")
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'gamma': [0, 0.1, 0.2]
        }
        
        xgb = XGBClassifier()
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_smote, y_train_smote)
        
        best_xgb = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        st.write("Best Parameters:", best_params)
        
        # Evaluate best model
        y_pred = best_xgb.predict(X_test_scaled)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix (Best XGBoost Model)")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        # Feature Importance
        st.subheader("Feature Importance (XGBoost)")
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(best_xgb, ax=ax)
        st.pyplot(fig)
        
        # SHAP Analysis
        st.subheader("SHAP Analysis")
        explainer = shap.TreeExplainer(best_xgb)
        shap_values = explainer.shap_values(X_test_scaled)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar")
        st.pyplot(fig)
        
    elif section == "Business Insights":
        st.title("Business Insights and Recommendations")
        
        st.markdown("""
        ### Key Findings:
        1. **High-income customers ($120K)** have significant attrition risk (17.3%)
        2. **Women** have 3% higher attrition rate than men
        3. **Customers with high transaction counts** are less likely to churn
        4. **Customers with few bank relationships** are more likely to churn
        5. **Customers who frequently contact the bank** may have unresolved issues
        
        ### Actionable Recommendations:
        
        **Boosting Transactions:**
        - Offer 5% cashback for transactions (minimum 3x/month)
        - Provide interactive transaction tips via email or social media
        - Offer relevant marketplace discounts
        
        **Strengthening Customer Relationships:**
        - Automatically upgrade Blue card holders to Silver/Gold for high transaction volumes (>$10k/month)
        - Offer high-interest savings products for credit-card-only customers
        
        **Improving Customer Service:**
        - Retrain CS teams for faster resolution (e.g., 24-hour deadline)
        - Review CRM data and conduct evaluations
        """)
        
        st.image("https://www.salesforce.com/content/dam/blogs/ca/Blog%20Posts/sales-and-service-are-now-one-open-graph.jpg", 
                width=600)
        
else:
    st.warning("Please upload the BankChurners.csv file to proceed with the analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Prasetyo Sukma Raharjo")
st.sidebar.markdown("Contact: prasetyo.sukmaraharjo@gmail.com")