# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset (replace with actual path to your dataset)
data = pd.read_csv('./insurance_claims.csv')

# Display basic info
print(data.info())
print(data.head())

# Define target variable and features
target = 'fraud_reported'
features = ['Age', 'Vehicle_Age', 'Annual_Mileage', 'Accident_History', 'Location', 'Vehicle_Type']

# Separate features and target
X = data[features]
y = data[target]

# Preprocessing: Handling categorical features & scaling numeric ones
categorical_features = ['Location', 'Vehicle_Type']
numeric_features = ['Age', 'Vehicle_Age', 'Annual_Mileage', 'Accident_History']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Random Forest Regressor pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the Random Forest model
rf_pipeline.fit(X_train, y_train)

# Predict on test data
rf_preds = rf_pipeline.predict(X_test)

# Evaluate the model using MAE and MSE
mae = mean_absolute_error(y_test, rf_preds)
mse = mean_squared_error(y_test, rf_preds)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')

# Feature Importance Visualization
feature_importance = rf_pipeline.named_steps['model'].feature_importances_
features_all = numeric_features + list(rf_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': features_all,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Random Forest Feature Importance')
plt.show()

# Prediction Comparison: Actual vs Predicted Claims
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=rf_preds)
plt.xlabel('Actual Claim Amount')
plt.ylabel('Predicted Claim Amount')
plt.title('Actual vs Predicted Car Insurance Claims')
plt.show()
