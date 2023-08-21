# Nigerian House Price Prediction
This repository contains the code and resources for an end-to-end Housing Price Prediction MLOps project. The project utilizes a variety of tools and technologies to ensure efficient and robust development, deployment, and monitoring of a machine learning model for predicting housing prices.

![show](images/House_prices.jpg)

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Problem Description](#problem-description)
- [Dataset Description](#dataset-description)
- [Modeling](#modeling)
- [MLOps Pipeline](#mlops-pipeline)
- [Workflow Orchestration](#workflow-orchestration)
- [Monitoring and Visualization](#monitoring-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict Nigerian housing prices using a machine learning model. It incorporates various DevOps and MLOps practices to ensure streamlined development, deployment, and monitoring of the model.


## Technologies Used

- ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white) **Visual Studio Code**: A powerful and versatile code editor with built-in debugging, version control, and an extensive extension ecosystem.

- ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) **Jupyter Notebook**: An interactive, web-based environment for data analysis and scientific computing that supports code, visualizations, and narrative text.

- ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white) **PostgreSQL**: A powerful open-source relational database management system known for its extensibility, reliability, and advanced features.

- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) **Python**: A widely-used high-level programming language known for its simplicity and readability, commonly used for data manipulation and machine learning.

- ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) **Pandas**: A Python library for data manipulation and analysis, providing data structures and functions to efficiently work with structured data.

- ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) **Matplotlib**: A comprehensive data visualization library in Python, used to create static, interactive, and animated visualizations.

- ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) **scikit-learn**: A machine learning library for Python that provides simple and efficient tools for data mining and data analysis.

- ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) **Flask**: A lightweight web application framework in Python, suitable for building web applications and APIs.

- ![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white) **MLflow**: An open-source platform for managing the end-to-end machine learning lifecycle, including experiment tracking, reproducibility, and deployment.

- ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) **Docker**: A platform for developing, shipping, and running applications in containers, ensuring consistency across various environments.

- ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) **Anaconda**: A distribution of Python and R programming languages for data science and machine learning, providing a variety of packages and tools.

- ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white) **Linux**: An open-source operating system kernel widely used for server environments, development, and hosting.

- ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white) **Amazon Web Services (AWS)**: A cloud computing platform offering a wide range of services for computing power, storage, and other functionalities.

- ![Grafana](https://img.shields.io/badge/grafana-%23F46800.svg?style=for-the-badge&logo=grafana&logoColor=white) **Grafana**: A monitoring and visualization tool used to track metrics, create dashboards, and gain insights from data.

- ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) **Git**: A distributed version control system used for tracking changes in code and collaborating with others.

- ![Pylint](https://img.shields.io/badge/pylint-%230A7ACA.svg?style=for-the-badge&logo=pylint&logoColor=white) Linter for ensuring code quality and adherence to coding standards.
- ![Black](https://img.shields.io/badge/black-%23000000.svg?style=for-the-badge&logo=black&logoColor=white) Code formatter for maintaining consistent code style.
- ![isort](https://img.shields.io/badge/isort-%EF9030.svg?style=for-the-badge&logo=isort&logoColor=white) Tool for sorting and formatting Python imports.
- ![Pre-commit](https://img.shields.io/badge/pre--commit-%23FAB040.svg?style=for-the-badge&logo=pre-commit&logoColor=white) Framework for managing and maintaining pre-commit hooks.


## Project Structure

The project has been structured with the following folders and files:

- `.github:` contains the CI/CD files (GitHub Actions)
- `config:` contains grafana config files
- `dashboards:` contains json format for monitoring dashboards
- `data:` dataset and test sample for testing the model
- `model:` full pipeline from preprocessing to prediction and monitoring using MLflow, Prefect, Grafana, Adminer, and docker-compose
- `notebooks:` EDA and Modeling performed at the beginning of the project to establish a baseline
- `tests:` unit tests
- `pyproject.toml:` linting and formatting
- `requirements.txt:` project requirements


## Getting Started

1. Clone the repository: `git clone https://github.com/yourusername/your-repo.git`
2. Set up your environment and install dependencies: `pip install -r requirements.txt`
3. Follow instructions in relevant sections below to run preprocessing, training, and deployment.

## Problem Description

The goal of this project is to develop a machine learning model that can accurately predict housing prices in Nigeria for various types of houses across all 36 states. The model aims to take into account features such as the type of house, location, bedroom size, bathroom size, parking space size, and other relevant factors to make accurate predictions. This predictive model can be a valuable tool for real estate professionals, homeowners, and potential buyers to estimate property values.

## Dataset Description

The dataset used for this project is a Nigeria House Price Dataset that covers all 36 states of the country. It is located in the `data/` directory. It contains the following columns:

- **ID**: A unique identifier for each property.
- **Type of House**: The type of the house, such as apartment, duplex, bungalow, etc.
- **Location**: The location of the property within a specific state.
- **Bedroom Size**: The number of bedrooms in the house.
- **Bathroom Size**: The number of bathrooms in the house.
- **Parking Space Size**: The size of the parking space available.
- **Price**: The target variable, representing the price of the property.

## Modeling

The model training process is defined in the `model_train.py` file. It involves loading the preprocessed data, splitting it into training and validation sets, training a machine learning model and saving the model to model registry. The trained model is saved in the `models/` directory.

## MLOps Pipeline

1. Data preprocessing and feature engineering.
2. Model training and evaluation.
3. Model versioning using MLflow.
4. Continuous Integration (CI) using GitHub Actions for code quality checks and tests.
5. Continuous Deployment (CD) using GitHub Actions to deploy the model in a containerized environment.
6. Workflow orchestration using Prefect to schedule and manage the entire pipeline.

## Workflow Orchestration

The Workflow Orchestration phase of this project involves managing and automating the various steps of the machine learning pipeline using Prefect Cloud. It ensures that data preprocessing, model training, and deployment occur seamlessly and efficiently. The [Prefect](https://www.prefect.io/) workflow orchestration tool is utilized to schedule, coordinate, and monitor these tasks.

visit [Prefect Cloud]("https://www.youtube.com/watch?v=y89Ww85EUdo&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK") to setup prefect cloud.

### Prefect Deployment

    ```
    prefect deployment build main.py:main_run \
      -n "main_pipeline" \
      -o "main_pipeline" \
      --apply
    ```

![show](images/deployment.png)


## Monitoring and Visualization

Grafana is used to monitor various metrics and insights related to the model's performance, data quality, and more. It provides real-time visualization of key performance indicators and helps in identifying anomalies and trends.

![show](images/monitoring.png)

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow the standard GitHub workflow: fork the repository, create a feature branch, make your changes, and submit a pull request.

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/gbotemiB)
[![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/_oluwagbotty)
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/emmanuel-bolarinwa/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gbotemibolarinwa@gmail.com)


## License

This project is licensed under the [MIT License](LICENSE).
