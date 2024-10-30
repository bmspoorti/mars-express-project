import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import json
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Wait for the preprocessing to complete
flag_file_path = '/flags/preprocessing_completed.flag'

print("Checking if preprocessing has completed...")

while not os.path.exists(flag_file_path):
    print("Preprocessing not completed yet. Waiting...")
    time.sleep(10)  # Wait for 10 seconds before checking again

print("Preprocessing completed. Starting model training.")

# Database connection parameters from environment variables
DB_HOST = os.environ.get('DB_HOST', 'db')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'mex_database')
DB_USER = os.environ.get('DB_USER', 'admin')
DB_PASS = os.environ.get('DB_PASS', '1234')

# Build the database URL
db_url = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create the SQLAlchemy engine
engine = create_engine(db_url)

table_name = 'preprocessed_train'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
merged_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)
print("total columns of preprocessed train:", len(merged_data.columns))

# Convert 'ut_ms' columns to datetime and align both datasets
merged_data['ut_ms'] = pd.to_datetime(merged_data['ut_ms'])

# Set the timestamp as index
merged_data.set_index('ut_ms', inplace=True)

# Define window parameters
window_size = pd.Timedelta(hours=7)  # 7 hours window
overlap_size = pd.Timedelta(hours=1)  # 1 hour overlap
step_size = window_size - overlap_size

# Prepare a scaler to normalize the features
scaler = StandardScaler()

# Create a function to train models and return the model with the best RMSE score
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'RandomForest': RandomForestRegressor(),
        'XGBoost': XGBRegressor(),
        'CatBoost': CatBoostRegressor(verbose=0)
    }
    best_model = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        print(f"Training model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"{name} RMSE: {rmse}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = (name, model)
    
    return best_model, best_rmse

# Create a directory for saving pickle files
output_dir = '/models/'
os.makedirs(output_dir, exist_ok=True)

# Initialize metadata dictionary
metadata = []

# Track previous best model
prev_best_model = None
current_window_data = []
file_counter = 1

print("Starting training over time windows...")

# Iterate over time windows
start_time = merged_data.index.min()
end_time = merged_data.index.max()

while start_time + window_size <= end_time:
    window_end = start_time + window_size
    print(f"\nProcessing window from {start_time} to {window_end}")
    window_data = merged_data.loc[start_time:window_end]
    
    # Check if 'NPWD2372' column exists
    if 'NPWD2372' not in window_data.columns:
        print("Warning: 'NPWD2372' column not found in data. Skipping this window.")
        start_time += step_size
        continue

    # Split into features (X) and target (y)
    X = window_data.drop(columns=['NPWD2372'])
    y = window_data['NPWD2372']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"X_train shape after scaling: {X_train.shape}")  # Number of rows and features in X_train
    print(f"X_test shape after scaling: {X_test.shape}")    # Number of rows and features in X_test
    print(f"y_train shape: {y_train.shape}")                # Number of rows in y_train
    print(f"y_test shape: {y_test.shape}")                  # Number of rows in y_test

    
    # Train the models and get the best model for this window
    best_model, best_rmse = train_models(X_train, y_train, X_test, y_test)
    print(f"Best model for this window: {best_model[0]} with RMSE: {best_rmse}")
    
    # Check if this window's model is the same as the previous window
    if prev_best_model and prev_best_model[0] == best_model[0]:
        # If the model is the same, extend the current window
        current_window_data.append((start_time, window_end, best_model[1], best_rmse))
    else:
        # If the model has changed, save the previous model data to a pickle file and update metadata
        if prev_best_model:
            # Save the pickle file
            model_file_path = os.path.join(output_dir, f'model_{prev_best_model[0]}_window_{file_counter}.pkl')
            with open(model_file_path, 'wb') as f:
                pickle.dump(prev_best_model[1], f)
            print(f"Saved model to {model_file_path}")
            # Update metadata with the start and end time for the model
            metadata.append({
                'model_name': prev_best_model[0],
                'file_path': model_file_path,
                'start_time': str(current_window_data[0][0]),
                'end_time': str(current_window_data[-1][1]),
                'rmse': best_rmse
            })
            file_counter += 1
        
        # Start a new window for the current model
        prev_best_model = best_model
        current_window_data = [(start_time, window_end, best_model[1], best_rmse)]
    
    # Shift the window by the step size (7 hours - 1 hour overlap)
    start_time += step_size

# After the loop ends, save the last model
if prev_best_model:
    model_file_path = os.path.join(output_dir, f'model_{prev_best_model[0]}_window_{file_counter}.pkl')
    with open(model_file_path, 'wb') as f:
        pickle.dump(prev_best_model[1], f)
    print(f"Saved final model to {model_file_path}")
    # Update metadata for the final model
    metadata.append({
        'model_name': prev_best_model[0],
        'file_path': model_file_path,
        'start_time': str(current_window_data[0][0]),
        'end_time': str(current_window_data[-1][1]),
        'rmse': best_rmse
    })

# Save metadata to a JSON file
metadata_file_path = os.path.join(output_dir, 'metadata.json')
with open(metadata_file_path, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"Saved metadata to {metadata_file_path}")

print(f"\nModel training and saving complete. All models are saved in the '{output_dir}' folder.")

# Create a completion flag file
flag_file_path = '/flags/model_training_completed.flag'
with open(flag_file_path, 'w') as flag_file:
    flag_file.write('Model training completed.')
print("Model training completed successfully.")

