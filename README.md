# Diabetes Detection Using XGBoost and ANNs

## Project Overview
This project focuses on predicting the onset of diabetes based on diagnostic measurements. It utilizes a combination of traditional statistical analysis, machine learning (XGBoost), and Deep Learning (Artificial Neural Networks) to create a robust classification system.

## Dataset
The project uses the `diabetes.csv` dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
* **Target Variable**: `Outcome` (0: Non-diabetic, 1: Diabetic)
* **Features**: 
    * Health metrics: Glucose, Blood Pressure, BMI, Insulin.
    * Demographic info: Age, Number of Pregnancies.
    * Genetic factors: Diabetes Pedigree Function.

## Workflow

### 1. Data Exploration & Visualization
* Statistical analysis of the distribution of features.
* Correlation heatmaps to identify relationships between variables like Age/Pregnancies and Glucose/Outcome.

### 2. Data Preprocessing
* **Cleaning**: Handling "hidden" missing values (zeros in columns like BloodPressure or BMI where zero is medically impossible).
* **Splitting**: Dividing data into Training and Testing sets.
* **Scaling**: Normalizing data to ensure the Neural Network stabilizes during training.

### 3. Model Architecture
The project implements three distinct approaches:
* **Logistic Regression**: Used as a baseline performance metric.
* **XGBoost**: An ensemble gradient boosting algorithm optimized for speed and accuracy on tabular data.
* **Artificial Neural Networks (ANN)**: A multi-layer deep learning model using:
    * **Dense Layers**: For feature extraction.
    * **Dropout Layers**: To prevent overfitting.
    * **Adam Optimizer**: For efficient weight updates.

### 4. Evaluation
Models are evaluated based on:
* **Accuracy Score**: Overall correctness.
* **Confusion Matrix**: Visualizing Type I and Type II errors.
* **Classification Report**: Precision, Recall, and F1-score for both classes.

## Installation & Requirements
To run this notebook, ensure you have the following installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
