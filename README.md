# Bank Marketing Subscription Prediction

## 1. Project Overview

This project uses machine learning to predict whether a customer is likely to subscribe to a bank's term deposit following a marketing campaign.

The project combines exploratory data analysis (EDA), data preprocessing, machine learning model comparison, and a Streamlit web application to support data-driven marketing decisions.

## 2. Business Objective

Bank marketing campaigns require time and resources to contact potential customers. Identifying customers who are more likely to subscribe can help marketing teams prioritize their outreach activities.

The main objectives are to:

- Analyze customer and campaign data.
- Identify patterns associated with term deposit subscriptions.
- Build and compare multiple classification models.
- Predict the likelihood of a customer subscribing.
- Recommend strategies to improve marketing campaign effectiveness.

## 3. Dataset Overview

The project uses the Portuguese Bank Marketing dataset.

**Dataset file:** `bank-additional-full.csv`

The dataset contains customer demographics, previous campaign information, economic indicators, and the outcome of the marketing campaign.

**Target variable:** `y`

- `yes` — The customer subscribed to a term deposit.
- `no` — The customer did not subscribe.

The dataset initially contained 41,188 records. After removing 12 duplicate records, 41,176 records remained for analysis.

### Data Preparation

The following preprocessing steps were used:

- Removed duplicate records.
- Checked for missing values.
- Examined the distribution of the target variable.
- Encoded categorical variables using one-hot encoding.
- Removed the `duration` feature from predictive modeling to avoid data leakage when making predictions before a call.
- Addressed class imbalance during model training using class weighting.

## 4. Exploratory Data Analysis

Exploratory data analysis was performed to understand customer characteristics, campaign outcomes, and relationships between features and subscription behavior.

The analysis included:

- Target variable distribution.
- Customer demographics.
- Job and education categories.
- Campaign contact methods.
- Previous campaign outcomes.
- Monthly campaign patterns.
- Economic indicators.
- Feature importance.

### Key Findings

- The dataset is imbalanced, with substantially more non-subscribers than subscribers.
- The `pdays` value of `999` represents customers who were not previously contacted in the relevant campaign data.
- Economic indicators, including `euribor3m`, `nr.employed`, and `emp.var.rate`, were among the most important features in the Random Forest model.
- Previous campaign outcome was also an important predictive feature.

## 5. Machine Learning Models

Three classification algorithms were trained and compared.

### Logistic Regression

A classification algorithm used as a baseline model to estimate the relationship between input features and subscription outcomes.

### Decision Tree Classifier

A tree-based algorithm that learns decision rules from customer and campaign characteristics.

### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve predictive performance.

## 6. Model Performance Comparison

The following results were obtained during model evaluation.

| Metric    | Logistic Regression | Decision Tree | Random Forest |
|---:       |---:                 |---:           |---:           |
| Accuracy  | 82.75%              | 84.82%        | 85.42%        |
| Precision | 35.5%               | 39.2%         | 40.6%         |
| Recall    | 64.8%               | 62.8%         | 63.9%         |
| F1-Score  | 0.458               | 0.483         | 0.497         |

### Best Performing Model

The Random Forest Classifier achieved the highest accuracy and F1-score among the three evaluated models.

Its performance was:

- Accuracy: 85.42%
- Precision: 40.6%
- Recall: 63.9%
- F1-score: 0.497

Although Random Forest performed best in this comparison, its precision indicates that many customers predicted as subscribers may not actually subscribe. Additional evaluation and threshold tuning would be appropriate before production deployment.

## 7. Streamlit Prediction Application

A Streamlit application was developed to provide an interactive interface for customer subscription prediction.

The application loads the trained Random Forest model and its corresponding feature columns to generate predictions from customer inputs.

### Application Features

- Interactive customer input form.
- Machine learning-based prediction.
- Trained model integration.
- Feature-column alignment for prediction inputs.
- Results displayed through a web interface.

## 8. Project Structure

```text
Bank_Marketing/
│
├── app.py
├── Portuguese Bank Marketing.ipynb
├── bank-additional-full.csv
├── bank_marketing_random_forest.pkl
├── model_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Jupyter Notebook
- Streamlit
- Git and GitHub

## 10. How to Run the Application

### Prerequisites

Install Python and Git, and ensure that the project files are available locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sunvilkumar26-cyber/Bank-Marketing-Prediction.git
```

### Step 2: Open the Project Folder

```bash
cd Bank-Marketing-Prediction
```

### Step 3: Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, use the local address displayed in the terminal.

## 11. Marketing Recommendations

Based on the analysis and model evaluation, the following actions may help improve campaign effectiveness.

1. **Prioritize Potential Subscribers:** Use model predictions to help prioritize customer outreach, while validating the predictions against actual campaign outcomes.

2. **Analyze Previous Campaign Results:** Examine previous campaign outcomes to identify customer segments and approaches associated with successful subscriptions.

3. **Evaluate Contact Strategies:** Compare campaign performance across contact methods and months to identify opportunities for improvement.

4. **Personalize Marketing Campaigns:** Use relevant customer and campaign characteristics to develop targeted outreach strategies.

5. **Monitor Model Performance:** Evaluate precision, recall, and F1-score on future campaign data, and adjust the prediction threshold according to campaign costs and business objectives.

6. **Validate Before Deployment:** Test predictions on new data and assess the financial impact of false positives and false negatives before using the model for operational decisions.

## 12. Challenges and Limitations

### Class Imbalance

The dataset contains substantially more non-subscribers than subscribers. Class weighting was used during model training to help address this imbalance.

### Categorical Features

Categorical variables were converted into numerical representations using one-hot encoding.

### Special Values in `pdays`

The value `999` has a special meaning and represents customers who were not previously contacted in the relevant campaign data.

### Data Leakage

The `duration` feature was excluded because call duration is not known before a call begins. Using it for pre-call targeting could lead to misleading evaluation results.

### Model Performance

Although Random Forest achieved the best overall results among the evaluated models, its precision remains limited. Further testing, threshold tuning, and business-impact analysis are needed before production use.

## 13. Future Improvements

- Evaluate additional classification algorithms.
- Perform hyperparameter tuning.
- Optimize the prediction threshold using business costs and benefits.
- Improve model interpretability.
- Evaluate performance across customer segments.
- Monitor predictions against actual campaign outcomes.
- Deploy the application to a public hosting platform.

## 14. Conclusion

This project demonstrates how machine learning can support bank marketing decisions by identifying customers who may be more likely to subscribe to a term deposit.

The Random Forest Classifier achieved the strongest results among the three evaluated models. Combined with exploratory data analysis and an interactive Streamlit application, the project provides a foundation for data-driven marketing campaign planning.

Further validation and performance improvements are recommended before operational deployment.