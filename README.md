# Machine Learning Projects

Automotive Pricing & Logistics Optimization
Overview
This repository showcases applied machine learning projects focused on pricing intelligence and logistics optimization, combining predictive modeling with real-world business impact.
The projects span automotive market analysis and a Saudi-based retail logistics case study, demonstrating end-to-end workflows from data preprocessing to financial impact evaluation.

# Projects Overview
1. Car Price Classification
Notebook: Cars_Classification.ipynb

This project categorizes vehicles into distinct market segments based on their technical specifications and pricing characteristics.

Objective

Classify cars into three market categories: Luxury, Mid-Range, and Budget

Key Features

Engine Capacity (CC)

Horsepower

Fuel Type

Manufacturer (Company Name)

Models Used

CatBoostClassifier – Achieved high recall through cross-validation, leveraging categorical feature handling

RandomForestClassifier – Used as a baseline model for performance comparison

Technologies

Scikit-learn

CatBoost

AutoGluon

Plotly (interactive visualizations)

2. Car Price Regression
Notebook: Cars_Regression.ipynb

Building on the classification task, this project focuses on predicting the exact market price of vehicles.

Objective

Predict car prices with high accuracy using advanced regression techniques

Methodology

Data Cleaning: Custom regex-based functions to extract numerical values from text fields (e.g., "3990 cc" → 3990)

Preprocessing: Log transformation of the target variable to address price skewness and improve model stability

Models Used

Weighted Ensemble (Voting Regressor):

Combines CatBoost and XGBoost with a 3:2 weight ratio to maximize the R² score

Random Forest Regressor:

Tuned for depth and number of estimators as a strong baseline model

3. The Shipping Crisis: Saudi Retail Case Study
Notebook: Shipping_Crisis_Template.ipynb

A strategic analytics and predictive modeling project for a Saudi-based e-commerce platform facing recurring delivery delays.

Scenario

Investigating why packages fail to reach customers on time

Determining whether delays are driven by logistics, product characteristics, or strategic decisions

Key Tasks

Root Cause Analysis:

Exploratory visualizations analyzing delay patterns across:

Warehouse_Zone

Mode_of_Shipment

Discount_Offered

Predictive Modeling:

Binary classification to predict shipment delays

Target variable: is_delayed

1 → Delayed

0 → On Time

Financial Impact Analysis

Includes a custom Saudi Riyal (SAR) impact function to quantify:

Cost of False Negatives (lost customer lifetime value)

Benefit of True Positives (customers retained through proactive intervention)

Results & Business Impact
Automotive Projects

Developed a reusable pipeline that standardizes raw car data from Kaggle

Delivered both market segmentation and continuous price prediction capabilities

# Retail Logistics Project

Created a decision-support framework to identify high-risk shipments

Enabled proactive interventions resulting in an estimated SAR 61,650.00 net savings

# Skills Demonstrated
Feature engineering and data cleaning

Categorical data modeling (CatBoost)

Ensemble learning and model evaluation

Cost-sensitive and impact-driven machine learning

Exploratory data analysis and visualization

Business-oriented ML decision making

# Requirements
The following libraries are required to run the notebooks:

pandas, numpy

scikit-learn

catboost, xgboost

autogluon

plotly

kagglehub (for dataset loading)

# How to Use
1. Clone the Repository
git clone https://github.com/Komail-Prog/week3-ml-baseline-system.git

2. Install Dependencies
pip install pandas scikit-learn catboost xgboost autogluon plotly kagglehub

3. Run the Notebooks
Open any .ipynb file using Jupyter Lab or VS Code to explore the data analysis, modeling, and results.
