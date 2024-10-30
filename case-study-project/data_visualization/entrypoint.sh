# data_visualization/entrypoint.sh

#!/bin/bash
set -e

# Wait for the database to be ready
./wait-for-it.sh --timeout=60 db:5432 -- echo "Database is up"

# Wait for model_testing to complete
while [ ! -f /flags/model_testing_completed.flag ]; do
  echo "Waiting for model_testing to complete..."
  sleep 5
done

# Run the data visualization script
streamlit run Dataviz.py --server.port 8501 --server.address 0.0.0.0
