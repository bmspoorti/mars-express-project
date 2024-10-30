
# Mars Express Orbiter Thermal Power Consumption Prediction

## Table of Contents

* Introduction
* System Architecture
* Technologies
* Getting Started
    * Clone the repository
    * Run the Application
* Services Description
* Project Structure
* Results

## Introduction

The Mars Express Orbiter Thermal Power Consumption Prediction project aims to predict the thermal power consumption (NPWD2372) of the Mars Express Orbiter using historical telemetry data. The project involves data ingestion, preprocessing, model training, testing, and visualization, all orchestrated using Docker and Docker Compose.

## System Architecture
![architecture_diagram_casestudy](https://github.com/user-attachments/assets/2a16de0e-1299-43a5-aba8-e09585d61214)

The project is designed with the following components:

* **Data Source**: Raw CSV files containing data from the Mars Express Orbiter mission.
* **PostgreSQL Database**: Stores ingested and preprocessed data.
* **Data Ingestion Service**: Reads CSV files and inserts the data into the PostgreSQL database.
* **Preprocessing Service**: Cleans and preprocesses the data for model training.
* **Model Training Service**: Trains machine learning models using the preprocessed data.
* **Model Testing Service**: Tests the trained models on test data and generates predictions.
* **Data Visualization** : Streamlit app to visualize data and predictions.
* **Docker and Docker Compose**: Used for containerization and orchestration of services.

## Technologies
The project utilizes the following technologies:

* **Docker**: Containerization platform for packaging services.
* **Docker Compose**: Tool for defining and running multi-container Docker applications.
* **PostgreSQL**: Relational database for storing ingested and preprocessed data.
* **Python**: Programming language used for scripting and data processing.
* **Pandas**: Data manipulation and analysis library.
* **NumPy**: Fundamental package for scientific computing.
* **scikit-learn**: Machine learning library for model training.
* **XGBoost**: Gradient boosting framework.
* **CatBoost**: Gradient boosting library with categorical features support.
* **SQLAlchemy**: Database toolkit and ORM for Python.
* **psycopg2**: PostgreSQL database adapter for Python.
* **Streamit**: Streamlit app to visualize data and predictions.

## Getting Started
### Prerequisites
* **Docker**: Ensure Docker is installed on your system.
* **Docker Compose**: Required to orchestrate multiple containers.
* **Git**: To clone the repository.
* **Data Files**: You need to have the required CSV data files placed in the appropriate directories as specified.

## 1. Clone the Repository

``` git clone https://github.com/Bhargav141922/mars-express-orbiter-thermal-power-consumption-prediction.git```

```cd mars-express-orbiter-thermal-power-consumption-prediction ```

## 2. Run the Application
Ensure you have your data files in place (see Project Structure for details).

Start the entire pipeline using Docker Compose:

```docker-compose up```

This command will pull the necessary Docker images from Docker Hub and start all the services defined in the ```docker-compose.yml``` file.

## Services Description
* Data Ingestion Service (```data_ingestion```)

    * Reads CSV files and inserts data into the PostgreSQL database.
    * Image: ```bhargavdeekshithvasu1997/mars_express_orbiter:data_ingestion```
* Preprocessing Service (```preprocessing```)

    * Cleans and preprocesses the data for model training.
    * Image: ```bhargavdeekshithvasu1997/mars_express_orbiter:preprocessing```
* Model Training Service (```model_training```)

    * Trains machine learning models using the preprocessed data.
    * Image: ```bhargavdeekshithvasu1997/mars_express_orbiter:model_training```
* Model Testing Service (```model_testing```)

    * Tests the trained models on test data and generates predictions.
    * Image: ```bhargavdeekshithvasu1997/mars_express_orbiter:model_testing```
* Data Visualization Service (```data_visualization```)
    * Visualises the solar aspect angles and the models suitable for a particular window of time.
    * Image: ```bhargavdeekshithvasu1997/mars_express_orbiter:data_visualization```
* PostgreSQL Database Service (```db```)

    * Stores ingested and preprocessed data.
    * Image: ```postgres:13```

## Project Structure

```
├── data/                      # Data directory (to be populated by the user)
│   ├── training/
│   └── testing/
├── models/                    # Directory to store trained models
├── results/                   # Directory to store predictions
├── data_ingestion/
│   └── insert_data.py         # Data ingestion script
├── preprocessing/
│   └── preprocessing.py       # Data preprocessing script
├── model_training/
│   └── training.py            # Model training script
├── model_testing/
│   └── testing.py             # Model testing script
├── data_visualization/
│   └── Dataviz.py 
├── docker-compose.yml         # Docker Compose configuration
└── README.md                  # Project README file
```

## Results

* Predictions are saved in the ```results/predictions.csv``` file.
* Trained models and metadata are stored in the ```models/``` directory.
"# mars-express-project" 
