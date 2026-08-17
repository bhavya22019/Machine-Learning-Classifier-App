# Machine Learning Classification Dashboard

## a. Problem Statement

The objective of this project is to build and compare multiple Machine Learning classification models using the Bank Marketing dataset. The application automatically preprocesses the dataset, trains multiple classification algorithms, evaluates their performance using standard classification metrics, and displays the results through an interactive Streamlit dashboard. Users can also upload their own CSV dataset for analysis.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| Dataset | Bank Marketing Dataset |
| Source | UCI Machine Learning Repository (UCI) |
| Number of Samples | 45,211 |
| Number of Features | 16 Input Features + 1 Target |
| Problem Type | Binary Classification |
| Target Variable | **y** (yes / no) |

The dataset contains customer information collected during direct marketing campaigns conducted by a Portuguese banking institution. The goal is to predict whether a customer will subscribe to a term deposit.

---

## c. GitHub Repository Link

**GitHub Repository**

https://github.com/bhavya22019/Machine-Learning-Classifier-App.git

**Live Streamlit Application**

https://machine-learning-classifier-app-ccwxrb4tffjxdjbdmmfs6q.streamlit.app/

---

## d. Models Used

The following Machine Learning models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|---------:|----:|----------:|-------:|---------:|----:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6450 | 0.3469 | 0.4511 | 0.4257 |
| Decision Tree | 0.8749 | 0.7021 | 0.4662 | 0.4764 | 0.4712 | 0.4004 |
| K-Nearest Neighbors (KNN) | 0.8963 | 0.8276 | 0.6007 | 0.3384 | 0.4329 | 0.3997 |
| Gaussian Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | **0.9046** | **0.9263** | **0.6526** | 0.3941 | **0.4915** | **0.4595** |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved high overall accuracy and precision. It performed well but had relatively lower recall for the positive class. |
| Decision Tree | Produced the lowest overall accuracy among the tree-based methods. It is simple and interpretable but showed comparatively lower overall performance. |
| K-Nearest Neighbors (KNN) | Delivered competitive accuracy with good precision, although recall remained relatively low compared to other models. |
| Gaussian Naive Bayes | Achieved the highest recall (0.5198), making it better at identifying positive cases, but its overall accuracy and precision were the lowest. |
| Random Forest (Ensemble) | Achieved the highest Accuracy, AUC, Precision, F1 Score, and MCC, making it the best-performing model overall on the Bank Marketing dataset. |
| **Overall Winner** | **Random Forest**, as it achieved the best overall performance across most evaluation metrics. |

---

## Streamlit Dashboard Features

- Automatic analysis of the Bank Marketing dataset
- Upload custom CSV datasets
- Automatic data preprocessing
- Multiple machine learning model training
- Model comparison table
- Accuracy comparison chart
- Classification reports
- Confusion matrices
- Download results as CSV

---

## Requirements

The project uses the following Python libraries:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- ucimlrepo

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Project Links

**GitHub Repository**

https://github.com/bhavya22019/Machine-Learning-Classifier-App.git

**Live Streamlit Application**

https://machine-learning-classifier-app-ccwxrb4tffjxdjbdmmfs6q.streamlit.app/

---

## Author

**Name:** M Bhavya Sri

**Student ID:** 2025AC05805
