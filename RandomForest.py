import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the training data
train = pd.read_csv(r"C:\Users\raghu\Downloads\datasetb2d9982 (1)\dataset\train.csv")

# Select the features and target variable
features = ['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'PRODUCT_TYPE_ID']
target = 'PRODUCT_LENGTH'

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train[features], train[target], test_size=0.2, random_state=0)

# Define a random forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate the root mean squared error (RMSE) on the validation set
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Validation RMSE: {rmse:.2f}')

# Load the test data
test = pd.read_csv('test.csv')

# Make predictions on the test set
test['PRODUCT_LENGTH'] = model.predict(test[features])

# Save the predictions to a CSV file
test[['PRODUCT_ID', 'PRODUCT_LENGTH']].to_csv('submission.csv', index=False)