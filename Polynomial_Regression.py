import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Step 1: Load the dataset
data = pd.read_csv('50_Startups.csv')

# Step 2: Preprocess the data
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]  # Features
y = data['Profit']  # Target variable

# Step 3: Create polynomial features
poly = PolynomialFeatures(degree=2)  # Using polynomial features of degree 2
X_poly = poly.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=10)

# Step 5: Fit the polynomial regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
# Calculate the range of the target variable
target_range = y.max() - y.min()
y_pred = model.predict(X_test)
print('\n')

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
mse_percentage = (mse / target_range) * 100
print("Mean Squared Error (Percentage):", mse_percentage)
print('\n')


# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
mae_percentage = (mae / target_range) * 100
print("Mean Absolute Error (MAE) Percentage:", mae_percentage)
print('\n')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
print('The accuracy of the model is : ',(model.score(X_test,y_test))*100,'%')
print('\n')


# Step 7: Optional - Visualize the results
plt.scatter(y_test, y_pred, c='red', label='Predicted')  # Predicted values in red
plt.scatter(y_test, y_test, c='blue', label='Actual')  # Actual values in blue
plt.xlabel("Profit (Actual)")
plt.ylabel("Profit (Predicted)")
plt.title("Actual vs. Predicted Profit")
plt.legend()
plt.show()
