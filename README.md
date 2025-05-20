# Prediction-of-Student-Academic-Performance-Using-Hybrid-Machine-Learning-Models

## Overview
This project aims to predict the academic performance of students using a hybrid machine learning approach. By combining multiple algorithms, the model seeks to improve prediction accuracy and provide insights into key factors affecting student success.

## Dataset
The dataset contains various features related to student demographics, academic history, and other relevant factors. These features are used to train and evaluate the machine learning models.

- Dataset file: `dataset.csv`
- Features include (example): student demographics, attendance, past grades, study hours, etc.

## Methodology
The project employs a hybrid machine learning model that integrates multiple algorithms to leverage their individual strengths. The pipeline includes:

1. **Data Preprocessing**  
   - Handling missing values  
   - Feature scaling and encoding  
   - Data splitting into training and testing sets  

2. **Model Selection**  
   - Algorithms used include  Random Forest, SVM, Gradient Boosting, Neural Networks and Logistic Regression 
   - Hybrid approach combines predictions or features from these models to enhance performance  

3. **Training and Evaluation**  
   - Models are trained on the processed data  
   - Evaluation metrics used: accuracy, precision, recall, F1-score, ROC-AUC, etc.  
   - Cross-validation for robust performance assessment  

4. **Prediction**  
   - The hybrid model generates final predictions on student academic outcomes  

## Results
- Summary of model performance on the test dataset.  
- Key insights on important features influencing student performance.  

## Usage
To run the notebook and reproduce the results:

1. Clone the repository  
2. Install required packages (e.g., `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`)  
3. Open and run the `Prediction-of-Student-Academic-Performance-Using-Hybrid-Machine-Learning-Models.ipynb` notebook  
4. Explore data preprocessing, model training, and evaluation steps  

## Requirements
- Python 3.x  
- Google Colab 
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, etc.
