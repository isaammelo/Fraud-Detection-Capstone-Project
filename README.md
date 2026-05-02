# Fraud-Detection-Project-Using-Machine-Learning
Machine learning project for detecting fraudulent online transactions using the IEEE-CIS Fraud Detection dataset.

---

## 1. Business Problem / Motivation

Credit card fraud is a serious problem in the financial industry. Large numbers of fraudulent transactions lead to financial loss and can reduce customer trust. At the same time, flagging normal transactions as fraud can frustrate users and hurt the overall experience.

This project focuses on building a machine learning model that can classify transactions as fraud or non-fraud using both transaction and identity data.

---

## 2. Project Overview

This project uses the IEEE-CIS Fraud Detection dataset to build a full machine learning pipeline. The process includes data cleaning, exploratory data analysis, preprocessing, model training, and evaluation.

Several models were tested, including Logistic Regression, Decision Tree, Random Forest, and XGBoost. Imbalance handling techniques such as Random Oversampling, SMOTE, and ADASYN were also applied.

XGBoost performed the best based on ROC-AUC and showed a strong balance between precision and recall.

---

## 3. Data

**Source:** IEEE-CIS Fraud Detection dataset (Kaggle)  
https://www.kaggle.com/competitions/ieee-fraud-detection/data

**Type:** Tabular transaction and identity data  

**Size:** Around 590,000 transactions from:
- `train_transaction.csv`
- `train_identity.csv`

The datasets were merged using `TransactionID`.

**Target Variable:**
- `isFraud` (0 = non-fraud, 1 = fraud)

**Class Distribution:**
- About 96.5% non-fraud  
- About 3.5% fraud (high imbalance)

**Key Features:**
- Transaction amount (`TransactionAmt`)  
- Card-related features (`card1–card6`)  
- Address features (`addr1`, `addr2`)  
- Behavioral features (`C1–C14`)  
- Time-related features (`D1–D15`)  
- Engineered features (`V1–V339`)  

**Note:** Dataset is not included due to Kaggle rules. Download it and place it in the `data/` folder.

---

## 4. Data Preprocessing

- Merged transaction and identity datasets using `TransactionID`  
- Dropped columns with more than 70% missing values  
- Filled missing numeric values with median  
- Filled categorical values with mode  
- Applied one-hot encoding for categorical variables  
- Split data into 80/20 train and test sets  
- Applied scaling for models like Logistic Regression  
- Tree-based models were trained without scaling  

---

## 5. Exploratory Data Analysis (EDA)

**Class Imbalance:**  
Only about 3.5% of transactions are fraud, which makes the problem challenging.

![Class Imbalance](images/class_imbalance.png)

**Transaction Amount Distribution:**  
Most transactions are small, with a few large outliers.

![Transaction Distribution](images/transaction_distribution.png)

**Observations:**
- Fraud cases are much fewer than non-fraud  
- Transaction amounts are not evenly distributed  
- The dataset requires models that can handle imbalance  

---

## 6. Modeling Approach

**Baseline Model:**
- Logistic Regression  

**Other Models:**
- Decision Tree  
- Random Forest  
- XGBoost  

**Imbalance Handling:**
- Random Oversampling  
- SMOTE  
- ADASYN  

These were used to improve fraud detection for the minority class.

---

## 7. Model Training

**Tools Used:**
- scikit-learn  
- xgboost  
- imbalanced-learn  
- shap  
- joblib  

**Process:**
- Split data into training and testing sets  
- Train models on training data  
- Evaluate using test data  
- Apply imbalance techniques to improve recall  
- Compare models using multiple metrics  

---

## 8. Results

**Metrics Used:**
- Recall → detects fraud cases  
- Precision → accuracy of fraud predictions  
- F1-score → balance between precision and recall  
- ROC-AUC → overall performance  

**Best Model: XGBoost**

![Confusion Matrix](images/confusion_matrix_xgboost.png)

**Results Summary:**
- Very strong at identifying non-fraud cases  
- Very few false positives  
- Good detection of fraud cases  

**Model Comparison:**
- Logistic Regression → lowest performance  
- Decision Tree → moderate  
- Random Forest → strong  
- XGBoost → best  

**Imbalance Techniques:**
- Improved recall  
- Increased false positives  
- XGBoost without heavy oversampling gave better balance  

---

## 9. Model Interpretation

**Feature Importance:**

![Feature Importance](images/xgboost_feature_importance.png)

Shows which features influenced the model the most.

**SHAP Summary:**

![SHAP Summary](images/shap_summary.png)

Explains how features affect predictions.

**Insights:**
- Transaction amount is a key factor  
- Behavioral features help detect patterns  
- Some features increase fraud probability, others reduce it  

---

## 10. Key Insights

- XGBoost handled complex patterns better than other models  
- Class imbalance had a big impact on results  
- Oversampling helped detect more fraud but added false positives  
- Important features include transaction amount and behavior-related variables  

**Business Impact:**

A model with higher recall can catch more fraudulent transactions, which helps reduce losses. Keeping precision reasonable helps avoid flagging too many normal transactions.

---

## 11. Conclusion

This project built a full pipeline for fraud detection using machine learning. Multiple models were tested, and XGBoost performed the best.

The results show that combining preprocessing, imbalance handling, and strong models can improve fraud detection performance.

---

## 12. Future Work

- Tune model parameters  
- Improve feature engineering  
- Try additional models  
- Deploy as an application or API  
- Monitor performance over time  

---

## 13. How to Run

1. Clone the repository
```bash
git clone https://github.com/isaammelo/Fraud-Detection-Capstone-Project.git
cd Fraud-Detection-Capstone-Project
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Download dataset
- Go to: https://www.kaggle.com/competitions/ieee-fraud-detection/data
- Download `train_transaction.csv` and `train_identity.csv`
- Place both files inside the `data/` folder
4. Run the notebook
```bash
jupyter notebook notebooks/Fraud_Detection_Capstone_Project_Code.ipynb
```
5. Load saved model if needed
```python
import joblib
model = joblib.load("models/xgb_fraud_model.pkl")
```
---

## 14. Repository Structure Explanation

- `README.md` → full explanation of the project
- `requirements.txt` → list of required Python libraries
- `data/` → location for dataset files (not included)
- `notebooks/` → main notebook with full pipeline
- `models/` → saved trained models
- `results/` → evaluation results and outputs
- `images/` → visualizations used in the README

---

## 15. Requirements

To install all dependencies:
```bash
pip install -r requirements.txt
```
