# 📈 Time Series Forecasting: Sensor Count

This project focuses on forecasting **sensor count over time** using a combination of classical machine learning models and deep learning techniques.

## 📌 Objective

To predict future sensor activity based on historical time series data, helping understand trends, anomalies, and system behavior over time.

## 🧠 Models Implemented

1. **Linear Regression** – a simple baseline for comparison  
2. **Random Forest Regressor** – a robust ensemble model  
3. **XGBoost Regressor** – gradient boosting-based model known for performance  
4. **LSTM** – deep learning model designed for sequence prediction

## 🔁 Workflow Overview

### 1. Data Preprocessing
- Time formatting and resampling
- Missing value imputation
- Feature engineering (lags)

### 2. Model Training
- Train-test split based on time
- Sequence preparation for LSTM (windowing)

### 3. Evaluation
- Metrics: RMSE, MAE, R²
- Visualization: actual vs predicted sensor counts
- Comparison across models

### 4. Forecasting
- Multi-step forecasts
- Visual diagnostics for future prediction intervals

