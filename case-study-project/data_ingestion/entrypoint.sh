#!/bin/bash
set -e

# Wait for the database to be ready
./wait-for-it.sh --timeout=60 db:5432 -- echo "Database is up"

# Run the data ingestion script
python insert_data.py

# Indicate completion by creating a flag file
touch /flags/data_ingestion_completed.flag
