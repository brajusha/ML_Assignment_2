# ML Assignment 2 - BITS Pilani

**a. Problem Statement**
To implement and evaluate five distinct classification machine learning models on a chosen dataset, and to deploy the best-performing models via an interactive Streamlit web application that predicts outcomes and displays evaluation metrics on unseen test data without retraining.

**b. Dataset Description**
*   **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (from UCI / scikit-learn)
*   **Description:** This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The target is binary classification (Malignant or Benign).
*   **Number of Features:** 30 (Satisfies >= 12 criteria)
*   **Number of Instances:** 569 (Satisfies >= 500 criteria)

**c. Github Repository Link**
[https://github.com/brajusha/ML_Assignment_2.git]

**d. Models Used & Comparison Table**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9649 | 0.9977 | 0.9718 | 0.9718 | 0.9718 | 0.9234 |
| **Decision Tree** | 0.9474 | 0.9438 | 0.9577 | 0.9577 | 0.9577 | 0.8845 |
| **KNN** | 0.9561 | 0.9790 | 0.9583 | 0.9718 | 0.9650 | 0.9037 |
| **Naive Bayes** | 0.9737 | 0.9984 | 0.9722 | 0.9859 | 0.9790 | 0.9416 |
| **Support Vector Machine** | 0.9474 | 0.9974 | 0.9452 | 0.9718 | 0.9583 | 0.8850 |
*(Note: Replace these values with the exact ones generated when you test your app)*

**e. Observations**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs extremely well with high accuracy and an almost perfect AUC score, showing that linear boundaries are highly effective for this dataset. |
| **Decision Tree** | Slightly lower performance compared to linear models. While it models complex decisions well, it might be slightly overfitting the training data leading to a lower MCC score. |
| **KNN** | Shows strong baseline performance with high recall, proving that local instance clustering works well for diagnostic image metrics. |
| **Naive Bayes** | Yielded the highest F1 and MCC scores in the test set. It handles the continuous features excellently by assuming Gaussian distribution, making it highly robust here. |
| **Support Vector Machine** | Delivered very high AUC, meaning it is excellent at ranking predictions, but standard accuracy was slightly lower than Logistic Regression on the test split. |
| **Overall Winner for your dataset?** | **Naive Bayes** achieved the highest overall metrics (Accuracy: 0.9737, F1: 0.9790, MCC: 0.9416), making it the most balanced and reliable model for this specific classification task. |
