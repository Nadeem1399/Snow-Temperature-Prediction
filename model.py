import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data from CSV
file_path = 'SNEX23_MAR23_SP_Forest_20230316_BCEF_DA375_data_temperature_v01.0.csv'
data = pd.read_csv(file_path)

# Extract features (Depth) and target variables (Temperature)
depth = data['Depth (cm)'].values.reshape(-1, 1)
temperature = data['Temperature (deg C)'].values.reshape(-1, 1)

# Scale the data using MinMaxScaler
scaler_depth = MinMaxScaler(feature_range=(0, 1))
scaler_temp = MinMaxScaler(feature_range=(0, 1))

depth_scaled = scaler_depth.fit_transform(depth)  # Scale depth
temperature_scaled = scaler_temp.fit_transform(temperature)  # Scale temperature

# Create the model
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Model for Snow Depth prediction
model_depth = create_model()
model_depth.fit(depth_scaled, depth_scaled, epochs=50, verbose=1)

# Model for Temperature prediction
model_temp = create_model()
model_temp.fit(depth_scaled, temperature_scaled, epochs=50, verbose=1)

# Predict the values for future depths (simulate future depths as a simple sequence)
future_depths = np.array([80, 90, 100, 110, 120]).reshape(-1, 1)  # Example future depths

# Scale the future depths
future_depths_scaled = scaler_depth.transform(future_depths)

# Predict the future snow depth and temperature
predicted_snow_depth_scaled = model_depth.predict(future_depths_scaled)
predicted_temperature_scaled = model_temp.predict(future_depths_scaled)

# Inverse the scaling for the predictions
predicted_snow_depth = scaler_depth.inverse_transform(predicted_snow_depth_scaled)
predicted_temperature = scaler_temp.inverse_transform(predicted_temperature_scaled)

# Output the predicted values to the console in the format you prefer
print("Predicted Snow Depth:")
print(predicted_snow_depth)

print("\nPredicted Temperature:")
print(predicted_temperature)
