# **MLOps Flight Price Prediction Project**

This project builds an end-to-end machine learning pipeline for predicting flight prices based on various features such as departure and arrival times, airline, routes, and more. It encompasses the full lifecycle of machine learning, from data ingestion and preprocessing to model training, hyperparameter tuning, and cloud deployment. The project is built with best practices for MLOps, including version control, continuous integration/continuous deployment (CI/CD), model tracking, and containerization.

The pipeline integrates various technologies like **AWS**, **Docker**, **DVC**, **MLflow**, and **GitHub Actions** to ensure scalability, reproducibility, and automation.

---

## **Table of Contents**

- [**MLOps Flight Price Prediction Project**](#mlops-flight-price-prediction-project)
  - [**Table of Contents**](#table-of-contents)
  - [**Project Overview**](#project-overview)
  - [**Technologies Used**](#technologies-used)
  - [**Architecture Overview**](#architecture-overview)
  - [**Installation Guide**](#installation-guide)
    - [1. Clone the Repository](#1-clone-the-repository)

---

## **Project Overview**

This project is designed to predict flight prices based on historical flight data. It uses various machine learning techniques and cloud services to create an automated, scalable pipeline. The primary steps of the pipeline include:

1. **Data Ingestion**: Fetch flight data from multiple sources (APIs, databases).
2. **Notebook Experiments** After fetch data after that i did Notebook experiments.
3. **Data Cleaning** Use All the data cleaning techniques.
4. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
5. **Feature Engineering**: Create new features and optimize them for the model.
6. **Model Training**: Train models like Random Forest and XGBoost using the processed data.
7. **Hyperparameter Tuning**: Tune model parameters to enhance performance.
8. **Model Evaluation**: Evaluate the models using metrics like RMSE, MAE, etc.
9. **Model Deployment**: Deploy the trained model using AWS EC2 and Docker for containerization.
10. **CI/CD Pipeline**: Automate the workflow with GitHub Actions for testing, building, and deployment.

The goal of this project is to showcase best practices in MLOps, including versioning of data and models, continuous deployment, and automated monitoring.

---

## **Technologies Used**

- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, MLflow, Hyperopt, FastAPI
- **Cloud Services**: AWS (EC2, ECR, S3)
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **MLFLOW**: Expreriments Tracking
- **Version Control**: DVC (for versioning datasets and models and Pipeline Tracking)
- **Other Tools**: Streamlit (for web interface)

---

## **Architecture Overview**

This project follows an **end-to-end MLOps architecture** consisting of the following components:

1. **Data Ingestion**:
   - Pulls data from external sources such as URL, APIs, CSV files, or databases.
   - This Project Pulls data from URL external sources.
2. **Notebook Experiments** 
   - After fetching the data, I first performed experiments in a Jupyter notebook, which included data  cleaning, exploratory data analysis (EDA), visualization, hypothesis testing, correlation analysis, multicollinearity check, feature engineering, model building, hyperparameter tuning, and model evaluation.
3. **Data Cleaning**
   - In this project, the data cleaning steps include dropping duplicate values, imputing missing values, and converting columns to the correct data types etc.
   
4. **Data Processing and Transformation**:
   - Transforms raw data into a clean format by handling missing values, scaling numerical features, and encoding categorical features.
   
5. **Feature Engineering**:
   - Generates new features that could improve model performance, including time-based features, categorical transformations, and feature interaction.

6. **Modeling**:
   - Trains various machine learning models (Random Forest, XGBoost, Adaboost, Bagging Ensmble, DessionTrees etc..) to predict flight prices.
   
7. **Hyperparameter Tuning**:
   - Uses tools like **Hyperopt** for automated hyperparameter search and optimization.

8. **Model Evaluation**:
   - Evaluates models using metrics such as RMSE, MAE, R^2, Adjasted R^2 etc., and selects the best-performing model.

9.  **Deployment**:
   - The final model is deployed as an API on **AWS EC2** using **Docker** for containerization.

10. **Monitoring and Experiment Tracking**:
   - **MLflow** is used to track experiments and monitor model performance.

11. **CI/CD Pipeline**:
   - GitHub Actions automates testing, model training, and deployment.

---


## **Installation Guide**

To set up the project locally, follow the steps below:

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/JahidHasan010/MLOps-Flight-Price-Prediction.git
cd MLOps-Flight-Price-Prediction
