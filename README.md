# Credit Card Fraud Detection

## Overview

This project develops a machine learning solution to detect fraudulent credit card transactions using Python, PySpark, SMOTE, and Random Forest classification.

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions. The goal is to accurately identify fraudulent activity while minimizing false positives.

---

## Business Problem

Financial institutions process millions of transactions daily. Fraudulent transactions can lead to significant financial losses and customer dissatisfaction.

This project aims to:

* Detect fraudulent transactions
* Handle severe class imbalance
* Improve fraud identification accuracy
* Evaluate model performance using industry-standard metrics

---

## Dataset

The project uses the Credit Card Fraud Detection dataset containing:

* 284,807 transactions
* 492 fraudulent transactions
* 31 features
* PCA-transformed variables (V1–V28)
* Time
* Amount
* Class (Fraud / Non-Fraud)

Due to dataset size and distribution restrictions, the dataset is not included in this repository.

---

## Technologies Used

* Python
* PySpark
* Pandas
* NumPy
* Scikit-learn
* SMOTE
* Random Forest Classifier
* Matplotlib
* Seaborn

---

## Methodology

1. Data Loading and Exploration
2. Data Cleaning
3. Missing Value Analysis
4. Feature Scaling
5. Class Imbalance Handling using SMOTE
6. Random Forest Model Training
7. ROC-AUC Evaluation
8. Classification Report Analysis
9. Confusion Matrix Visualization

---

## Model Evaluation

Performance was evaluated using:

* ROC-AUC Score
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve

These metrics are more appropriate than accuracy because fraudulent transactions represent less than 1% of all observations.

---

## Repository Structure

```text
credit-card-fraud-detection/

├── notebooks/
│   └── fraud_detection.ipynb

├── reports/
│   ├── Detailed_Credit_Card_Fraud_Detection_Phase_I.pdf
│   └── Phase-II-report.docx

├── data/

├── images/

└── README.md
```

---

## Key Learnings

* Handling highly imbalanced datasets using SMOTE
* Fraud detection using supervised machine learning
* Model evaluation beyond simple accuracy metrics
* Building scalable data processing workflows with PySpark
* Visualizing classification performance for business stakeholders

---

## Author

**Samved Thimmasarthy**

Master of Science in Business Analytics

University of Texas at Arlington
