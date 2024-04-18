import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('50_Startups.csv')

# Step 2: Preprocess the data
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]  # Features
y = data['Profit']  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Step 4: Initialize and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Step 6: Calculate Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Calculate the range of the target variable
target_range = y.max() - y.min()

# Calculate errors in number and percentage
mse_percentage = (mse / target_range) * 100
rmse_percentage = (rmse / target_range) * 100
mae_percentage = (mae / target_range) * 100

print('\n')
print("Mean Squared Error (MSE):", mse)
print("Mean Squared Error (MSE) Percentage:", mse_percentage)

print('\n')
print("Root Mean Squared Error (RMSE):", rmse)
print("Root Mean Squared Error (RMSE) Percentage:", rmse_percentage)

print('\n')
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Error (MAE) Percentage:", mae_percentage)
print('\n')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
print('The accuracy of the model is : ',(model.score(X_test,y_test))*100,'%')
print('\n')


plt.scatter(y_test, y_pred, c='red', label='Predicted')  # Predicted values in red
plt.scatter(y_test, y_test, c='blue', label='Actual')  # Actual values in blue
plt.xlabel("Profit (Actual)")
plt.ylabel("Profit (Predicted)")
plt.title("Actual vs. Predicted Profit")
plt.legend()
plt.show()