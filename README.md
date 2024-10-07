# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load the dataset, define features (Age, Mileage, Horsepower), target (Price), and split into training and testing sets.
2. Feature Scaling: Standardize features using StandardScaler since regularization methods are sensitive to input scales.
3. Model Training: Train Ridge, Lasso, and ElasticNet models, each using regularization to prevent overfitting, and calculate the MSE for each.
4. Evaluation & Visualization: Compare the actual vs predicted car prices for each model visually and numerically using Mean Squared Error (MSE).

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: HAMZA FAROOQUE
RegisterNumber:  212223040054
*/
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Sample data - car attributes (Age, Mileage, Horsepower) and Price
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mileage': [5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000],
    'Horsepower': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    'Price': [20000, 18500, 17500, 16500, 15500, 14500, 13500, 12500, 11500, 10500]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Feature variables (Age, Mileage, Horsepower)
X = df[['Age', 'Mileage', 'Horsepower']]

# Target variable (Price)
y = df['Price']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for Ridge, Lasso, ElasticNet)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Ridge Regression -----
ridge_regressor = Ridge(alpha=1.0)
ridge_regressor.fit(X_train_scaled, y_train)
ridge_pred = ridge_regressor.predict(X_test_scaled)
ridge_mse = mean_squared_error(y_test, ridge_pred)
print(f"Ridge Regression MSE: {ridge_mse}")

# ----- Lasso Regression -----
lasso_regressor = Lasso(alpha=1.0)
lasso_regressor.fit(X_train_scaled, y_train)
lasso_pred = lasso_regressor.predict(X_test_scaled)
lasso_mse = mean_squared_error(y_test, lasso_pred)
print(f"Lasso Regression MSE: {lasso_mse}")

# ----- ElasticNet Regression -----
elasticnet_regressor = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio=0.5 is a mix of Ridge and Lasso
elasticnet_regressor.fit(X_train_scaled, y_train)
elasticnet_pred = elasticnet_regressor.predict(X_test_scaled)
elasticnet_mse = mean_squared_error(y_test, elasticnet_pred)
print(f"ElasticNet Regression MSE: {elasticnet_mse}")

# ----- Visualization -----
# Plot the actual vs predicted prices for each model
plt.figure(figsize=(10, 6))

plt.scatter(y_test, ridge_pred, color='blue', label='Ridge Predictions', alpha=0.6)
plt.scatter(y_test, lasso_pred, color='green', label='Lasso Predictions', alpha=0.6)
plt.scatter(y_test, elasticnet_pred, color='purple', label='ElasticNet Predictions', alpha=0.6)

# Perfect prediction line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Prediction Line')

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices (Ridge, Lasso, ElasticNet)')
plt.legend()
plt.show()

```

## Output:
```
Ridge Regression MSE: 3999.4723543400787
Lasso Regression MSE: 21990.04315101876
ElasticNet Regression MSE: 151499.71944940859
```

![image](https://github.com/user-attachments/assets/f9cbfa28-cce2-496c-9dc3-db82c692e122)


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
