#!/bin/bash
set -e

# Wait for the database to be ready
./wait-for-it.sh --timeout=60 db:5432 -- echo "Database is up"

# Wait for data_ingestion to complete
while [ ! -f /flags/data_ingestion_completed.flag ]; do
  echo "Waiting for data_ingestion to complete..."
  sleep 5
done

# Run the preprocessing script
python preprocessing.py

# Indicate completion by creating a flag file
touch /flags/preprocessing_completed.flag
