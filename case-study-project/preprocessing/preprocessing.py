import pandas as pd
import numpy as np
import os
import time  
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")

# Wait for the data ingestion to complete
flag_file_path = '/flags/data_ingestion_completed.flag'

print("Checking if data ingestion has completed...")

while not os.path.exists(flag_file_path):
    print("Data ingestion not completed yet. Waiting...")
    time.sleep(10)  # Wait for 10 seconds before checking again

print("Data ingestion completed. Starting preprocessing.")

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
table_name = 'SAAF_train'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
saaf_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Convert ut_ms to a timestamp in datetime format
saaf_data['ut_ms'] = pd.to_datetime(saaf_data['ut_ms'], unit='ms')

# Set the timestamp as the index
saaf_data.set_index('ut_ms', inplace=True)

# Drop the original ut_ms column as it's no longer needed
#saaf_data.drop(columns=['ut_ms'], inplace=True)

# Resample the data to 15-minute intervals, using mean values for the aggregation
saaf_data_resampled = saaf_data.resample('15T').mean() 

# Create cosine transformations of the angles sa, sx, sy, sz
saaf_data_resampled['cos_sa'] = np.cos(np.radians(saaf_data_resampled['sa']))
saaf_data_resampled['cos_sx'] = np.cos(np.radians(saaf_data_resampled['sx']))
saaf_data_resampled['cos_sy'] = np.cos(np.radians(saaf_data_resampled['sy']))
saaf_data_resampled['cos_sz'] = np.cos(np.radians(saaf_data_resampled['sz']))

# Display the first few rows of the preprocessed data
print(saaf_data_resampled.head())


# Define the list of all known groups
all_groups = ['AXXX', 'AAAA', 'ASEQ', 'ATTT', 'APSF', 'AMMM', 'MOCS', 'PENS', 'PENE', 'MPER',
              'MOCE', 'MAPO', 'SCMN', 'AOOO', 'ASSS', 'AHHH', 'ASXX', 'AVVV', 'ATMB', 'PPNS',
              'PPNE', 'APWF', 'UPBS', 'UPBE', 'PDNS', 'PDNE', 'UDBS', 'UDBE', 'AACF', 'OBCP',
              'ADMC', 'Trig', 'DISA']

# Table name to query
table_name = 'DMOP_train'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
dmop_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Load the DMOP CSV file (replace the file path with your actual path)
# dmop_data = pd.read_csv('context--2010-07-10_2012-05-27--dmop.csv')

# Convert 'ut_ms' to a timestamp in datetime format
dmop_data['ut_ms'] = pd.to_datetime(dmop_data['ut_ms'], unit='ms')

# Set the timestamp as the index
dmop_data.set_index('ut_ms', inplace=True)

# Extract the first 4 characters as the subsystem group and the rest as the command
dmop_data['subsystem_group'] = dmop_data['subsystem'].str[:4]
dmop_data['subsystem_command'] = dmop_data['subsystem'].str[4:]

# Resample the data to 15-minute intervals, counting occurrences of each subsystem group
dmop_data_resampled = dmop_data.resample('15T').subsystem_group.value_counts().unstack(fill_value=0)

# Ensure that all groups are present in the resampled dataframe
for group in all_groups:
    if group not in dmop_data_resampled.columns:
        dmop_data_resampled[group] = 0

# Rearrange columns to match the order of the group list
dmop_data_resampled = dmop_data_resampled[all_groups]

# Define the fixed top 5 pairs manually (as per your given data)
top_5_pairs = [('AMMM', 'ATTT'), ('APSF', 'ATTT'), ('AMMM', 'APSF'), ('AAAA', 'ATTT'), ('ASSS', 'ATTT')]

# Create a new DataFrame to store the counts of the top 5 pairs in each 15-minute interval
top_5_pairs_counts = pd.DataFrame(index=dmop_data_resampled.index)

# Iterate over the predefined top 5 pairs and calculate their occurrences in each time interval
for pair in top_5_pairs:
    # For each pair, check if both subsystems were active in the interval
    top_5_pairs_counts[f'{pair[0]} & {pair[1]}'] = dmop_data_resampled.apply(
        lambda row: 1 if row[pair[0]] > 0 and row[pair[1]] > 0 else 0, axis=1
    )

# Merge the resampled data and the top 5 pairs counts
dmop_data = pd.concat([dmop_data_resampled, top_5_pairs_counts], axis=1)

# Save the final merged data to a CSV file
print(dmop_data.head())


# Predefined list of types
point_types = ['EARTH', 'SLEW', 'ACROSS_TRACK', 'MAINTENANCE', 'NADIR', 'INERTIAL',
               'RADIO_SCIENCE', 'WARMUP', 'D1PVMC', 'SPECULAR', 'D4PNPO', 'D3POCM', 
               'D2PLND', 'D7PLTS', 'D8PLTP', 'D5PPHB', 'SPOT', 'D9PSPO']

# Table name to query
table_name = 'FTL_train'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
flt_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# # Load the FTL data
# ftl_file_path = 'context--2010-07-10_2012-05-27--ftl.csv'  # Replace with your correct path
# ftl_df = pd.read_csv(ftl_file_path)

# Convert 'utb_ms' from milliseconds to datetime format and rename the column to 'ut_ms'
flt_data['ut_ms'] = pd.to_datetime(flt_data['utb_ms'], unit='ms')
flt_data.drop(columns=['ute_ms', 'utb_ms'], inplace=True)

# One-hot encoding function using the predefined list
def get_ohe(example, column_names):
    # Create a binary array indicating which column corresponds to the 'example' type
    return np.array([1 if col == example else 0 for col in column_names])

# Apply one-hot encoding to the 'type' column
print('One-hot encoding columns...')
ohe_point_type_cols = ['is_{}'.format(pt.lower()) for pt in point_types]
flt_data[ohe_point_type_cols] = flt_data['type'].apply(lambda x: pd.Series(get_ohe(x, point_types)))

# Drop the 'flagcomms' column as requested
flt_data.drop(columns=['flagcomms','type'], inplace=True)

# Set the 'ut_ms' datetime column as the index
flt_data.set_index('ut_ms', inplace=True)

# Select only numeric columns before resampling
numeric_columns = flt_data.select_dtypes(include=[np.number])

# Resample the data to 15-minute intervals and sum the values
ftl_df_resampled = numeric_columns.resample('15T').sum().fillna(0.0)

# Display the processed data
print(ftl_df_resampled.head())


# Table name to query
table_name = 'EVTF_train'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
evtf_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Load the EVTF CSV file
# evtf_file_path = 'context--2010-07-10_2012-05-27--evtf.csv'  # Replace with your correct file path
# evtf_data = pd.read_csv(evtf_file_path)

# Step 1: Convert ut_ms to a timestamp
evtf_data['ut_ms'] = pd.to_datetime(evtf_data['ut_ms'], unit='ms')

# Step 2: Extract relevant words from the description column
keywords = [
    'LOS', 'AOS', 'PHO_PENUMBRA_START', 'PHO_PENUMBRA_END', 'PHO_UMBRA_START', 'PHO_UMBRA_END',
    'MAR_PENUMBRA_START', 'MAR_PENUMBRA_END', 'MAR_UMBRA_START', 'MAR_UMBRA_END',
    'OCC_PHOBOS_START', 'OCC_PHOBOS_END', 'OCC_DEIMOS_START', 'OCC_DEIMOS_END',
    'DEI_PENUMBRA_START', 'DEI_PENUMBRA_END', 'ASCEND', 'DESCEND', 'PERICENTER', 'APOCENTER'
]

# Step 3: Create a column for each keyword and count occurrences in the description
for keyword in keywords:
    evtf_data[keyword] = evtf_data['description'].apply(lambda x: 1 if keyword in x else 0)

# Step 4: Set the timestamp as the index
evtf_data.set_index('ut_ms', inplace=True)

# Step 5: Align the timestamps to the nearest lower 15-minute mark (e.g., 00:00:00, 00:15:00, etc.)
evtf_data.index = evtf_data.index.floor('15T')

# Step 6: Resample the data into 15-minute intervals, summing the occurrences of each event
evtf_resampled = evtf_data[keywords].resample('15T').sum()

# Display the preprocessed and resampled data
print(evtf_resampled.head())

# Table name to query
table_name = 'LTDATA_train'  # Replace with your actual table name

# Load data from the database into a DataFrame
ltdata_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Step 1: Convert ut_ms to a timestamp in datetime format
ltdata_data['ut_ms'] = pd.to_datetime(ltdata_data['ut_ms'], unit='ms')

# Step 2: Set the timestamp as the index and drop the original ut_ms column
ltdata_data.set_index('ut_ms', inplace=True)
# ltdata_data.drop(columns=['ut_ms'], inplace=True)

# Step 3: Resample the data into 15-minute intervals, forward filling missing values
ltdata_resampled = ltdata_data.resample('15T').ffill()

# Display the resampled data (optional)
print(ltdata_resampled.head())


# Calculate the energy_received feature using the formula
p_max = 200000000000000000  # given value of p_max

energy_rece = pd.DataFrame(index=saaf_data_resampled.index)

# For each angle (sa, sx, sy, sz) in the SAAF file, we calculate the energy_received
for angle in ['sa', 'sx', 'sy', 'sz']:
    energy_rece[f'energy_received_{angle}'] = (p_max * saaf_data_resampled[f'cos_{angle}']) / (ltdata_resampled['sunmars_km'] ** 2)

saaf_data_resampled.drop(columns=['cos_sa', 'cos_sx', 'cos_sy', 'cos_sz'], inplace=True)

# Display the resulting dataframe with the new feature
print(energy_rece.head())


# power_data = pd.read_csv('power--2010-07-10_2012-05-27.csv')
# Table name to query
table_name = 'POWER_train'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
power_data = pd.read_sql_query(f'SELECT ut_ms, "NPWD2372" FROM "{table_name}"', con=engine)


power_data['ut_ms'] = pd.to_datetime(power_data['ut_ms'], unit='ms')
power_data = power_data.set_index('ut_ms')

power_data = power_data.resample('15T').mean().interpolate()
power_data.head()

# power_data.to_csv('Test_Power_Satellite.csv')

# Merge all datasets on the timestamp
data = pd.merge(saaf_data_resampled, energy_rece, on='ut_ms', how='outer')
data = pd.merge(data, dmop_data, on='ut_ms', how='outer')
data = pd.merge(data, evtf_resampled, on='ut_ms', how='outer')
data = pd.merge(data, ftl_df_resampled, on='ut_ms', how='outer')
data = pd.merge(data, ltdata_resampled, on='ut_ms',  how='outer')
data = pd.merge(data, power_data, on='ut_ms',  how='outer')

# Sort by timestamp
data = data.sort_values('ut_ms')

data = data.fillna(method='ffill').fillna(0)
# print(f"Columns in training data after merging power_data: {data.columns.tolist()}")


### Inserting the Preprocessed Data into the Database ###

print("Inserting the preprocessed data into the database...")
data.to_sql('preprocessed_train', engine, if_exists='replace', index=True, method='multi')
print("Data inserted into table 'preprocessed_train' successfully.")

# data.to_csv('Preprocess_Feature_MEX_2010.CSV')

# #TESTING

# # Database connection parameters from environment variables
# DB_HOST = os.environ.get('DB_HOST', 'db')
# DB_PORT = os.environ.get('DB_PORT', '5432')
# DB_NAME = os.environ.get('DB_NAME', 'mex_database')
# DB_USER = os.environ.get('DB_USER', 'admin')
# DB_PASS = os.environ.get('DB_PASS', '1234')

# # Build the database URL
# db_url = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# # Create the SQLAlchemy engine
# engine = create_engine(db_url)

# Table name to query
table_name = 'SAAF_test'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
saaf_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Convert ut_ms to a timestamp in datetime format
saaf_data['ut_ms'] = pd.to_datetime(saaf_data['ut_ms'], unit='ms')

# Set the timestamp as the index
saaf_data.set_index('ut_ms', inplace=True)

# Drop the original ut_ms column as it's no longer needed
#saaf_data.drop(columns=['ut_ms'], inplace=True)

# Resample the data to 15-minute intervals, using mean values for the aggregation
saaf_data_resampled = saaf_data.resample('15T').mean() 

# Create cosine transformations of the angles sa, sx, sy, sz
saaf_data_resampled['cos_sa'] = np.cos(np.radians(saaf_data_resampled['sa']))
saaf_data_resampled['cos_sx'] = np.cos(np.radians(saaf_data_resampled['sx']))
saaf_data_resampled['cos_sy'] = np.cos(np.radians(saaf_data_resampled['sy']))
saaf_data_resampled['cos_sz'] = np.cos(np.radians(saaf_data_resampled['sz']))

# Display the first few rows of the preprocessed data
print(saaf_data_resampled.head())


# Define the list of all known groups
all_groups = ['AXXX', 'AAAA', 'ASEQ', 'ATTT', 'APSF', 'AMMM', 'MOCS', 'PENS', 'PENE', 'MPER',
              'MOCE', 'MAPO', 'SCMN', 'AOOO', 'ASSS', 'AHHH', 'ASXX', 'AVVV', 'ATMB', 'PPNS',
              'PPNE', 'APWF', 'UPBS', 'UPBE', 'PDNS', 'PDNE', 'UDBS', 'UDBE', 'AACF', 'OBCP',
              'ADMC', 'Trig', 'DISA']

# Table name to query
table_name = 'DMOP_test'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
dmop_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Load the DMOP CSV file (replace the file path with your actual path)
# dmop_data = pd.read_csv('context--2010-07-10_2012-05-27--dmop.csv')

# Convert 'ut_ms' to a timestamp in datetime format
dmop_data['ut_ms'] = pd.to_datetime(dmop_data['ut_ms'], unit='ms')

# Set the timestamp as the index
dmop_data.set_index('ut_ms', inplace=True)

# Extract the first 4 characters as the subsystem group and the rest as the command
dmop_data['subsystem_group'] = dmop_data['subsystem'].str[:4]
dmop_data['subsystem_command'] = dmop_data['subsystem'].str[4:]

# Resample the data to 15-minute intervals, counting occurrences of each subsystem group
dmop_data_resampled = dmop_data.resample('15T').subsystem_group.value_counts().unstack(fill_value=0)

# Ensure that all groups are present in the resampled dataframe
for group in all_groups:
    if group not in dmop_data_resampled.columns:
        dmop_data_resampled[group] = 0

# Rearrange columns to match the order of the group list
dmop_data_resampled = dmop_data_resampled[all_groups]

# Define the fixed top 5 pairs manually (as per your given data)
top_5_pairs = [('AMMM', 'ATTT'), ('APSF', 'ATTT'), ('AMMM', 'APSF'), ('AAAA', 'ATTT'), ('ASSS', 'ATTT')]

# Create a new DataFrame to store the counts of the top 5 pairs in each 15-minute interval
top_5_pairs_counts = pd.DataFrame(index=dmop_data_resampled.index)

# Iterate over the predefined top 5 pairs and calculate their occurrences in each time interval
for pair in top_5_pairs:
    # For each pair, check if both subsystems were active in the interval
    top_5_pairs_counts[f'{pair[0]} & {pair[1]}'] = dmop_data_resampled.apply(
        lambda row: 1 if row[pair[0]] > 0 and row[pair[1]] > 0 else 0, axis=1
    )

# Merge the resampled data and the top 5 pairs counts
dmop_data = pd.concat([dmop_data_resampled, top_5_pairs_counts], axis=1)

# Save the final merged data to a CSV file
print(dmop_data.head())


# Predefined list of types
point_types = ['EARTH', 'SLEW', 'ACROSS_TRACK', 'MAINTENANCE', 'NADIR', 'INERTIAL',
               'RADIO_SCIENCE', 'WARMUP', 'D1PVMC', 'SPECULAR', 'D4PNPO', 'D3POCM', 
               'D2PLND', 'D7PLTS', 'D8PLTP', 'D5PPHB', 'SPOT', 'D9PSPO']

# Table name to query
table_name = 'FTL_test'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
flt_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# # Load the FTL data
# ftl_file_path = 'context--2010-07-10_2012-05-27--ftl.csv'  # Replace with your correct path
# ftl_df = pd.read_csv(ftl_file_path)

# Convert 'utb_ms' from milliseconds to datetime format and rename the column to 'ut_ms'
flt_data['ut_ms'] = pd.to_datetime(flt_data['utb_ms'], unit='ms')
flt_data.drop(columns=['ute_ms', 'utb_ms'], inplace=True)

# One-hot encoding function using the predefined list
def get_ohe(example, column_names):
    # Create a binary array indicating which column corresponds to the 'example' type
    return np.array([1 if col == example else 0 for col in column_names])

# Apply one-hot encoding to the 'type' column
print('One-hot encoding columns...')
ohe_point_type_cols = ['is_{}'.format(pt.lower()) for pt in point_types]
flt_data[ohe_point_type_cols] = flt_data['type'].apply(lambda x: pd.Series(get_ohe(x, point_types)))

# Drop the 'flagcomms' column as requested
flt_data.drop(columns=['flagcomms','type'], inplace=True)

# Set the 'ut_ms' datetime column as the index
flt_data.set_index('ut_ms', inplace=True)

# Select only numeric columns before resampling
numeric_columns = flt_data.select_dtypes(include=[np.number])

# Resample the data to 15-minute intervals and sum the values
ftl_df_resampled = numeric_columns.resample('15T').sum().fillna(0.0)

# Display the processed data
print(ftl_df_resampled.head())


# Table name to query
table_name = 'EVTF_test'  # Replace with your actual table name

# Load data from the database into a DataFrame
print(f"Loading data from table '{table_name}'...")
evtf_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Load the EVTF CSV file
# evtf_file_path = 'context--2010-07-10_2012-05-27--evtf.csv'  # Replace with your correct file path
# evtf_data = pd.read_csv(evtf_file_path)

# Step 1: Convert ut_ms to a timestamp
evtf_data['ut_ms'] = pd.to_datetime(evtf_data['ut_ms'], unit='ms')

# Step 2: Extract relevant words from the description column
keywords = [
    'LOS', 'AOS', 'PHO_PENUMBRA_START', 'PHO_PENUMBRA_END', 'PHO_UMBRA_START', 'PHO_UMBRA_END',
    'MAR_PENUMBRA_START', 'MAR_PENUMBRA_END', 'MAR_UMBRA_START', 'MAR_UMBRA_END',
    'OCC_PHOBOS_START', 'OCC_PHOBOS_END', 'OCC_DEIMOS_START', 'OCC_DEIMOS_END',
    'DEI_PENUMBRA_START', 'DEI_PENUMBRA_END', 'ASCEND', 'DESCEND', 'PERICENTER', 'APOCENTER'
]

# Step 3: Create a column for each keyword and count occurrences in the description
for keyword in keywords:
    evtf_data[keyword] = evtf_data['description'].apply(lambda x: 1 if keyword in x else 0)

# Step 4: Set the timestamp as the index
evtf_data.set_index('ut_ms', inplace=True)

# Step 5: Align the timestamps to the nearest lower 15-minute mark (e.g., 00:00:00, 00:15:00, etc.)
evtf_data.index = evtf_data.index.floor('15T')

# Step 6: Resample the data into 15-minute intervals, summing the occurrences of each event
evtf_resampled = evtf_data[keywords].resample('15T').sum()

# Display the preprocessed and resampled data
print(evtf_resampled.head())

# Table name to query
table_name = 'LTDATA_test'  # Replace with your actual table name

# Load data from the database into a DataFrame
ltdata_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# Step 1: Convert ut_ms to a timestamp in datetime format
ltdata_data['ut_ms'] = pd.to_datetime(ltdata_data['ut_ms'], unit='ms')

# Step 2: Set the timestamp as the index and drop the original ut_ms column
ltdata_data.set_index('ut_ms', inplace=True)
# ltdata_data.drop(columns=['ut_ms'], inplace=True)

# Step 3: Resample the data into 15-minute intervals, forward filling missing values
ltdata_resampled = ltdata_data.resample('15T').ffill()

# Display the resampled data (optional)
print(ltdata_resampled.head())


# Calculate the energy_received feature using the formula
p_max = 200000000000000000  # given value of p_max

energy_rece = pd.DataFrame(index=saaf_data_resampled.index)

# For each angle (sa, sx, sy, sz) in the SAAF file, we calculate the energy_received
for angle in ['sa', 'sx', 'sy', 'sz']:
    energy_rece[f'energy_received_{angle}'] = (p_max * saaf_data_resampled[f'cos_{angle}']) / (ltdata_resampled['sunmars_km'] ** 2)

saaf_data_resampled.drop(columns=['cos_sa', 'cos_sx', 'cos_sy', 'cos_sz'], inplace=True)

# Display the resulting dataframe with the new feature
print(energy_rece.head())


# power_data = pd.read_csv('power--2010-07-10_2012-05-27.csv')
# Table name to query
# table_name = 'POWER_train'  # Replace with your actual table name

# # Load data from the database into a DataFrame
# print(f"Loading data from table '{table_name}'...")
# power_data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con=engine)

# power_data['ut_ms'] = pd.to_datetime(power_data['ut_ms'], unit='ms')
# power_data = power_data.set_index('ut_ms')

# power_data = power_data.resample('15T').mean().interpolate()
# power_data.head()

# power_data.to_csv('Test_Power_Satellite.csv')

# Merge all datasets on the timestamp
data = pd.merge(saaf_data_resampled, energy_rece, on='ut_ms', how='outer')
data = pd.merge(data, dmop_data, on='ut_ms', how='outer')
data = pd.merge(data, evtf_resampled, on='ut_ms', how='outer')
data = pd.merge(data, ftl_df_resampled, on='ut_ms', how='outer')
data = pd.merge(data, ltdata_resampled, on='ut_ms',  how='outer')

# Sort by timestamp
data = data.sort_values('ut_ms')

data = data.fillna(method='ffill').fillna(0)
# print(f"Columns in testing data after merging power_data: {data.columns.tolist()}")


print("Inserting the preprocessed data into the database...")
data.to_sql('preprocessed_test', engine, if_exists='replace', index=True, method='multi')
print("Data inserted into table 'preprocessed_test' successfully.")

# Create a completion flag file
flag_file_path = '/flags/preprocessing_completed.flag'
with open(flag_file_path, 'w') as flag_file:
    flag_file.write('Preprocessing completed.')
