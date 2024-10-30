# data_ingestion/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure scripts are executable
RUN chmod +x entrypoint.sh wait-for-it.sh

# Set the entrypoint
ENTRYPOINT ["./entrypoint.sh"]
