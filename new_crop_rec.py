import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load dataset (Ensure dataset contains all required fields)
data = pd.read_csv("crop_recommend.csv")

# Features for both models
X = data[['temperature', 'humidity', 'ph', 'rainfall']]

# Target variables
y_crop = data['label']  # Crop label (classification)
y_npk = data[['N', 'P', 'K']]  # NPK values (regression)

# Split datasets
X_train, X_test, y_crop_train, y_crop_test = train_test_split(X, y_crop, test_size=0.2, random_state=42)
_, _, y_npk_train, y_npk_test = train_test_split(X, y_npk, test_size=0.2, random_state=42)

# Normalize features
ms = MinMaxScaler()
X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)

# **Train Crop Prediction Model (Classifier)**
crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
crop_model.fit(X_train_scaled, y_crop_train)

# **Train NPK Prediction Model (Regressor)**
npk_model = RandomForestRegressor(n_estimators=100, random_state=42)
npk_model.fit(X_train_scaled, y_npk_train)

# Save models and scaler
pickle.dump(crop_model, open('model.pkl', 'wb'))  # Save crop classifier model
pickle.dump(npk_model, open('npk_model.pkl', 'wb'))  # Save NPK regression model
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))  # Save MinMaxScaler

print("✅ Models Trained and Saved Successfully!")
