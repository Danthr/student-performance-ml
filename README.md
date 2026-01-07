# Student Performance Prediction using Machine Learning

## Overview
This project analyzes and predicts student academic performance using supervised and unsupervised machine learning techniques. The goal is to understand factors influencing performance and build predictive models using both regression and classification approaches.

## Dataset
The project uses the Student Performance dataset, which contains demographic, academic, and social attributes of secondary school students.

### Target Variables
- Regression: Final grade (G3)
- Classification: Pass / Fail (G3 ≥ 10)

## Project Workflow
1. Problem understanding and dataset exploration
2. Exploratory Data Analysis (EDA)
3. Data preprocessing and feature selection
4. Linear Regression implemented from scratch
5. Logistic Regression implemented from scratch
6. Model comparison using sklearn (Logistic Regression, Decision Tree, Random Forest)
7. Unsupervised learning using K-Means clustering and PCA

## Models Used
- Linear Regression (from scratch)
- Logistic Regression (from scratch)
- Logistic Regression (sklearn)
- Decision Tree Classifier
- Random Forest Classifier
- K-Means Clustering
- Principal Component Analysis (PCA)

## Evaluation Metrics
- RMSE for regression
- Accuracy and confusion matrix for classification
- Cluster interpretation for unsupervised learning

## Key Learnings
- Importance of EDA before modeling
- Feature scaling for gradient descent-based models
- Bias–variance tradeoff in model selection
- Difference between supervised and unsupervised learning
- How ensemble methods improve generalization

## Tools & Libraries
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

## How to Run
1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Run notebooks in order from `01_` to `06_`

## Future Improvements
- Add regularization techniques
- Tune hyperparameters
- Include more features
- Deploy as a web application
