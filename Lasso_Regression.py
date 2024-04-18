import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv('50_Startups.csv')

# Step 2: Preprocess the data
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]  # Features
y = data['Profit']  # Target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Step 4: Initialize and train the Lasso regression model
alpha = 1.0  # Regularization strength, you can adjust this parameter
model = Lasso(alpha=alpha)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\n')
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)

# Calculate the total error and its percentage
total_error = np.sum((y_test - y_pred) ** 2)
print("Total Error:", total_error)
print('\n')


total_error_percentage = (total_error / np.sum((y_test - np.mean(y_test)) ** 2)) * 100
print("Total Error Percentage:", total_error_percentage)
print('\n')

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
print('The accuracy of the model is : ',(model.score(X_test,y_test))*100,'%')
print('\n')


# Step 6: Plot actual vs. predicted values
plt.scatter(y_test, y_pred, c='red', label='Predicted')  # Predicted values in red
plt.scatter(y_test, y_test, c='blue', label='Actual')  # Actual values in blue
plt.xlabel("Profit (Actual)")
plt.ylabel("Profit (Predicted)")
plt.title("Actual vs. Predicted Profit")
plt.legend()
plt.show()
