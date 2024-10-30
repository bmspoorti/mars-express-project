#!/bin/bash
set -e

# Wait for the database to be ready
./wait-for-it.sh --timeout=60 db:5432 -- echo "Database is up"

# Wait for preprocessing to complete
while [ ! -f /flags/preprocessing_completed.flag ]; do
  echo "Waiting for preprocessing to complete..."
  sleep 5
done

# Run the model training script
python training.py

# Indicate completion by creating a flag file
touch /flags/model_training_completed.flag
