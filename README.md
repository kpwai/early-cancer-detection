# Early Cancer Detection Using the Breast Cancer Wisconsin (Diagnostic) Dataset

## Project Overview
This project aims to develop a machine learning pipeline for the early detection of breast cancer, classifying tumors as **malignant** or **benign**. The model is built using the Breast Cancer Wisconsin (Diagnostic) dataset.

## Objective
- To classify breast cancer tumors as malignant or benign using a machine learning model.
- Build a pipeline to preprocess the data, train models, and optimize hyperparameters.

## Dataset
The project uses the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). This dataset consists of features that describe characteristics of the cell nuclei present in breast cancer biopsies.

## Techniques Used

### 1. **Data Preprocessing**
- **Encoding**: The categorical target variable (diagnosis) is encoded as binary (0 for benign, 1 for malignant).
- **Feature Scaling**: Numerical features are standardized using `StandardScaler` from Scikit-learn.

### 2. **Modeling**
- **Random Forest Classifier**: A Random Forest model is used to classify the tumors, taking into account multiple decision trees to make the final prediction.

### 3. **Model Tuning**
- **Grid Search**: To optimize the performance of the model, hyperparameters are tuned using Grid Search with cross-validation.

## Installation
Clone this repository and install the required dependencies using the following commands:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Running the Project
Run the following command to execute the model:
```bash
python main.py
```
