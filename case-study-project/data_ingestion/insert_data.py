import psycopg2
import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore")

def get_connection():
    # Database connection details from environment variables
    try:
        conn = psycopg2.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', '5432'),
            database=os.environ.get('DB_NAME', 'mex_database'),
            user=os.environ.get('DB_USER', 'admin'),
            password=os.environ.get('DB_PASS', '1234')
        )
        print("Database connection established.")
        return conn
    except psycopg2.Error as e:
        print("Unable to connect to the database.")
        print(e)
        sys.exit(1)

def create_table(cur, table_name, columns):
    # Drops the table if it exists
    drop_table_sql = f'DROP TABLE IF EXISTS "{table_name}";'
    cur.execute(drop_table_sql)
    print(f"Table '{table_name}' dropped (if it existed).")

    # Creates a table with the given columns
    columns_sql = ', '.join([f'"{col_name}" {data_type}' for col_name, data_type in columns.items()])
    create_table_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql});'
    cur.execute(create_table_sql)
    print(f"Table '{table_name}' created or already exists.")

def insert_data(cur, table_name, df):
    # Inserts data into the table
    cols = ', '.join([f'"{col}"' for col in df.columns])
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders})'
    data_tuples = [tuple(row) for row in df.itertuples(index=False, name=None)]
    cur.executemany(insert_sql, data_tuples)
    print(f"Data inserted into table '{table_name}' successfully.")

def process_files(conn, csv_files, table_name):
    print(f"Processing files for table '{table_name}'...")
    dfs = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"Reading file: {csv_file}")
            df = pd.read_csv(csv_file)
            dfs.append(df)
        else:
            print(f"File not found: {csv_file}")
    if dfs:
        # Concatenate all dataframes
        df_combined = pd.concat(dfs, ignore_index=True)
        cur = conn.cursor()

        # Map data types
        dtype_mapping = {
            'int64': 'BIGINT',
            'float64': 'DOUBLE PRECISION',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'TEXT'
        }
        columns = {col: dtype_mapping.get(str(dtype), 'TEXT') for col, dtype in df_combined.dtypes.items()}

        create_table(cur, table_name, columns)
        insert_data(cur, table_name, df_combined)

        conn.commit()
        cur.close()
    else:
        print(f"No files found for table '{table_name}'.")

def main():
    conn = get_connection()

    # Process training data
    training_files = {
        'DMOP_train': [
            'data/training/context--2008-08-22_2010-07-10--dmop.csv',
            'data/training/context--2010-07-10_2012-05-27--dmop.csv',
            'data/training/context--2012-05-27_2014-04-14--dmop.csv'
        ],
        'SAAF_train': [
            'data/training/context--2008-08-22_2010-07-10--saaf.csv',
            'data/training/context--2010-07-10_2012-05-27--saaf.csv',
            'data/training/context--2012-05-27_2014-04-14--saaf.csv'
        ],
        'EVTF_train': [
            'data/training/context--2008-08-22_2010-07-10--evtf.csv',
            'data/training/context--2010-07-10_2012-05-27--evtf.csv',
            'data/training/context--2012-05-27_2014-04-14--evtf.csv'
        ],
        'FTL_train': [
            'data/training/context--2008-08-22_2010-07-10--ftl.csv',
            'data/training/context--2010-07-10_2012-05-27--ftl.csv',
            'data/training/context--2012-05-27_2014-04-14--ftl.csv'
        ],
        'LTDATA_train': [
            'data/training/context--2008-08-22_2010-07-10--ltdata.csv',
            'data/training/context--2010-07-10_2012-05-27--ltdata.csv',
            'data/training/context--2012-05-27_2014-04-14--ltdata.csv'
        ],
        'POWER_train': [
            'data/training/power--2008-08-22_2010-07-10.csv',
            'data/training/power--2010-07-10_2012-05-27.csv',
            'data/training/power--2012-05-27_2014-04-14.csv'
        ],
    }

    for table_name, csv_files in training_files.items():
        process_files(conn, csv_files, table_name)

    # Process testing data
    testing_files = {
        'DMOP_test': [
            'data/testing/context--2014-04-14_2016-03-01--dmop.csv'
        ],
        'SAAF_test': [
            'data/testing/context--2014-04-14_2016-03-01--saaf.csv'
        ],
        'EVTF_test': [
            'data/testing/context--2014-04-14_2016-03-01--evtf.csv'
        ],
        'FTL_test': [
            'data/testing/context--2014-04-14_2016-03-01--ftl.csv'
        ],
        'LTDATA_test': [
            'data/testing/context--2014-04-14_2016-03-01--ltdata.csv'
        ],
        'POWER_test': [
            'data/testing/power--2014-04-14_2016-03-01.csv'
        ],
    }

    for table_name, csv_files in testing_files.items():
        process_files(conn, csv_files, table_name)

    conn.close()
    print("Data ingestion completed.")

    # Create a completion flag file
    flag_file_path = '/flags/data_ingestion_completed.flag'
    with open(flag_file_path, 'w') as flag_file:
        flag_file.write('Data ingestion completed.')

if __name__ == '__main__':
    main()
