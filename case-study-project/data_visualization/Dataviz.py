# data_visualization/Dataviz.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
from sqlalchemy import create_engine

# Load the data from the database
@st.cache_data
def load_data():
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

    # Table name to query
    table_name = 'preprocessed_train'

    # Load data from the database into a DataFrame
    st.write(f"Loading data from table '{table_name}'...")
    data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)
    data['ut_ms'] = pd.to_datetime(data['ut_ms'])
    return data

data = load_data()

# Function to load predictions and metadata
@st.cache_data
def load_predictions():
    predictions_file = '/results/predictions.csv'  # Ensure this path matches the mounted volume
    predictions_data = pd.read_csv(predictions_file)
    predictions_data['timestamp'] = pd.to_datetime(predictions_data['timestamp'])
    return predictions_data

@st.cache_data
def load_metadata():
    metadata_file = '/models/metadata.json'  # Ensure this path matches the mounted volume
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata

# Load the predictions and metadata
predictions_data = load_predictions()
metadata = load_metadata()

# Sidebar for filters
st.sidebar.title("Filters")

# Date input for start and end dates for telemetry data
start_date = st.sidebar.date_input("Start Date for Telemetry", data['ut_ms'].min())
end_date = st.sidebar.date_input("End Date for Telemetry", data['ut_ms'].max())

# Dropdown menu for selecting parameter: sa, sx, sy, sz
parameter = st.sidebar.selectbox("Select Solar Aspect Angle", ['sa', 'sx', 'sy', 'sz'])

# Filter data based on the date input
filtered_data = data[(data['ut_ms'] >= pd.to_datetime(start_date)) & (data['ut_ms'] <= pd.to_datetime(end_date))]

# Title and subtitle
st.title("🌌 Satellite Anomaly Detection and Explanation")

st.markdown("""
Welcome to the **MEX Spacecraft Data Dashboard**! 
Use the filters on the left sidebar to explore telemetry data, energy received, commands, and model behavior.
""")

# Line chart for the selected parameter over time
st.header(f"Energy Received by {parameter.upper()} Over Time")
st.markdown("""
_This chart shows the energy received by the spacecraft over time based on the selected solar aspect angle._
Hover over the graph to view precise values.
""")

fig1, ax1 = plt.subplots()
ax1.plot(filtered_data['ut_ms'], filtered_data[f'energy_received_{parameter}'], label=f'Energy Received ({parameter.upper()})')
ax1.set_xlabel('Date')
ax1.set_ylabel(f'Energy Received ({parameter.upper()})')
ax1.legend()
st.pyplot(fig1)

# Commands over time
st.header("Top 10 Operational Commands Over Time")
st.markdown("""
_This graph shows the top 10 most frequent commands issued to the spacecraft over time. 
You can observe how these commands were distributed within the selected date range._
""")

# Select only the command columns (assuming they are from index 9 to 49)
command_columns = data.columns[9:49]

# Get the top 10 commands with the highest counts
command_sums = filtered_data[command_columns].sum().sort_values(ascending=False).head(10)
top_10_commands = command_sums.index

# Create a plot with 'ut_ms' on the x-axis and the top 10 command columns as legends
fig2, ax2 = plt.subplots(figsize=(10, 6))

for command in top_10_commands:
    ax2.plot(filtered_data['ut_ms'], filtered_data[command], label=command)

# Set plot labels and title
ax2.set_title('Top 10 Operational Commands Over Time')
ax2.set_xlabel('Date (ut_ms)')
ax2.set_ylabel('Command Values')

# Add legend
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Tight layout to avoid clipping
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig2)

# Satellite Model Behaviour
st.header("Satellite Model Behaviour")
st.markdown("""
_This section highlights the behavior of the satellite models over time. 
You can visualize the predicted power along with model change points for different machine learning models._
""")

# Date input for start and end dates in predictions data
start_date_pred = st.sidebar.date_input("Start Date for Model Behavior", predictions_data['timestamp'].min().date())
end_date_pred = st.sidebar.date_input("End Date for Model Behavior", predictions_data['timestamp'].max().date())

# Filter the predictions data based on the date input
filtered_predictions = predictions_data[(predictions_data['timestamp'] >= pd.to_datetime(start_date_pred)) & 
                                        (predictions_data['timestamp'] <= pd.to_datetime(end_date_pred))]

# Function to remove the year from timestamps (Martian day)
def to_martian_day(timestamp):
    return timestamp.replace(year=2000)  # Use a placeholder year to normalize

# Apply the Martian day conversion
filtered_predictions['martian_day'] = filtered_predictions['timestamp'].apply(to_martian_day)

# Set up the plot for predicted power
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the predicted values
ax.plot(filtered_predictions['martian_day'], filtered_predictions['predicted_value'], label='Predicted', color='blue', linestyle='--', linewidth=2)

# Define a color map for each model
model_colors = {
    'XGBoost': 'green',
    'RandomForest': 'red',
    'CatBoost': 'purple'
}

# Loop over the metadata to mark where model behavior changes
for entry in metadata:
    model_name = entry['model_name']
    model_start_time = pd.to_datetime(entry['start_time'])
    model_end_time = pd.to_datetime(entry['end_time'])
    
    # Adjust the start and end times to "Martian day" by ignoring the year
    martian_start = to_martian_day(model_start_time)
    martian_end = to_martian_day(model_end_time)
    
    # Mark model change with a vertical line using the corresponding color
    ax.axvline(x=martian_start, color=model_colors.get(model_name, 'black'), linestyle='--', label=f'{model_name} Change')

# Final plot adjustments
ax.set_title('Predicted Power with Model Change Points (Ignoring Year)')
ax.set_xlabel('Time')
ax.set_ylabel('Predicted Power (NPWD2372)')

# Adjust x-axis to reduce clutter
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  

# Keep x-axis labels straight (no rotation)
plt.xticks(rotation=0)

# Ensure the legend only shows one entry per model
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

# Display the plot in Streamlit
st.pyplot(fig)

# Customize the UI for presentation
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #FCA510;
    }
    .stMetric {
        color: #1f77b4;
    }
    .css-1v3fvcr p {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.write("**Developed by:** [Team Satellite] | **Data Source:** MEX Spacecraft")
