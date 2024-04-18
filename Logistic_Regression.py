# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report


# df = pd.read_csv('50_Startups.csv')
# df.head()

# X = df[['R&D Spend','Administration','Marketing Spend']]
# Y = df[['Profit']]

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# model = LogisticRegression()

# model.fit(X_train, Y_train)

# Y_pred = model.predict(X_test)
# # accuracy = accuracy_score(Y_test, Y_pred)
# # print("Accuracy:", accuracy)

# # print(classification_report(Y_test, Y_pred))

# # print(sum(Y_pred.tolist() == Y_test/len(Y_test)))


import pandas as pd

def check_continuous_variables(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Get the data types of each column
    dtypes = df.dtypes
    
    # Check if any column has a numeric data type
    continuous_variables = (dtypes == 'float64') | (dtypes == 'int64')
    
    # Get the names of the columns containing continuous variables
    continuous_columns = dtypes[continuous_variables].index.tolist()
    
    return continuous_variables.any(), continuous_columns

# Example usage
csv_file_path = '50_Startups.csv'
contains_continuous, continuous_columns = check_continuous_variables(csv_file_path)

if contains_continuous:
    print("The CSV file contains continuous variables in the following columns:", continuous_columns)
else:
    print("The CSV file does not contain continuous variables.")
