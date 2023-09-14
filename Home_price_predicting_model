import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Data Collection: Load GDP Growth data
gdp_data = pd.read_csv("C://Users/Vijay/Downloads/gdp_data.csv") # Replace with your actual GDP data source

# Data Preprocessing
# Assuming "DATE" column is in datetime format, otherwise, convert it to datetime
gdp_data['DATE'] = pd.to_datetime(gdp_data['DATE'])

# Set "DATE" as the index
gdp_data.set_index('DATE', inplace=True)

# Handle missing values (you can choose a different strategy if needed)
gdp_data = gdp_data.ffill()  # Forward-fill missing values

# Load the S&P Case-Shiller Home Price Index data
home_price_data = pd.read_csv("C:/Users/Vijay/Downloads/CSUSHPISA.csv")
home_price_data['DATE'] = pd.to_datetime(home_price_data['DATE'])
home_price_data.set_index('DATE', inplace=True)

# Merge GDP data with S&P Case-Shiller Index data based on a common date index
merged_data = home_price_data.join(gdp_data)

# Define the target variable (S&P Case-Shiller Index)
y = merged_data['CSUSHPISA']

# Define the features (GDP Growth in this example)
X = merged_data[['GDP']]  # Replace with other factors as needed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an imputer
imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', or 'most_frequent'

# Fit and transform the imputer on your training data
X_train_imputed = imputer.fit_transform(X_train)

# Replace X_train with the imputed data
X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)

# Remove rows with missing values in X_train and adjust y_train accordingly
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]  # Adjust y_train accordingly

# Apply the same imputer transformation to X_test
X_test_imputed = imputer.transform(X_test)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Visualize the results, e.g., plotting actual vs. predicted home prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Home Prices")
plt.ylabel("Predicted Home Prices")
plt.title("Actual vs. Predicted Home Prices")
plt.show()
