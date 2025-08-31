# data_science-assignment
# Data Science Assignment - Predictive Maintenance and Fraud Detection

## Overview
This repository contains solutions for two data science problems:
1. Predictive Maintenance in Manufacturing (Regression)
2. Fraud Detection in Banking (Classification & Clustering)

## Problem 1: Predictive Maintenance
### Approach
- Used Random Forest Regressor to address overfitting
- Applied feature engineering (interaction terms, normalization)
- Evaluated using RMSE, R², and cross-validation

### Results
- RMSE: 158.97
- R² Score: -0.2005
- Cross-validated RMSE:  147.60

## Problem 2: Fraud Detection
### Approach
- K-Means clustering for anomaly detection in unlabeled data
- Naïve Bayes classification for labeled data
- Feature engineering (time categories, amount ratios)
- Evaluated using F1-score and cross-validation

### Results
- F1 Score: [value]
- Cross-validated F1: [value]
- Potential fraud transactions identified: [value]

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
