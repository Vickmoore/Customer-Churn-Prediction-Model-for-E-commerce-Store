# Customer Churn Prediction Model for E-commerce Store

## Overview

This project implements a machine learning model to predict customer churn for an e-commerce store based on historical customer data. The goal is to identify customers likely to stop purchasing so that proactive retention strategies can be applied.

## Technologies Used

- Python
- Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib

## Setup Instructions

### Prerequisites

- Python 3.x installed
- Required Python libraries can be installed using pip. You can do this by running:
  pip install pandas numpy scikit-learn matplotlib

````

# Running the Model

a. Clone the repository or download the files to your local machine.
b. Open the Jupyter Notebook (churn_prediction_model.ipynb).
c. Load your customer dataset. Ensure that the dataset is in the correct format as specified in the Data Preparation section.
d. Run the cells in the notebook step by step to execute the model training and prediction.

# Interpreting the Output

a. The model outputs a probability score for each customer, indicating the likelihood of churn.
b. You can analyze the coefficients or feature importances to understand which features have the most influence on churn prediction.
c. Use visualizations to gain insights into the data and model performance.

# Key Features Influencing Churn

The following features have been identified as key influencers of customer churn based on the model's coefficients:
a. Purchase Frequency
b. Amount Spent
c. Time Since Last Purchase
d. Age
e. Customer Service Interactions

### 2. Basic Report

**Churn Prediction Model Report**

---

### Customer Churn Prediction Model Overview

This report outlines the workings of the customer churn prediction model designed for an e-commerce store. The model uses historical customer data to predict which customers are likely to stop purchasing, enabling proactive retention strategies.

### Model Functionality

The model is a binary classification system, utilizing either Logistic Regression or Random Forest to determine whether a customer will churn (1) or not (0). The main steps involved in building the model are as follows:

1. **Data Preparation**:

   - The dataset is cleaned and preprocessed, including handling missing values and encoding categorical variables.
   - Key features are engineered to represent customer behavior effectively.

2. **Model Training**:

   - The model is trained on a portion of the dataset (training set) and evaluated on a separate portion (test set).
   - The evaluation metrics used include Accuracy, Precision, Recall, and AUC-ROC to assess performance.

3. **Feature Importance**:
   - The importance of features is analyzed using coefficients from Logistic Regression or feature importances from Random Forest.
   - This helps identify which features most significantly influence churn.

### Key Features Influencing Churn

Based on the model, the following features have been identified as significant influencers of churn:

- **Purchase Frequency**: The frequency at which a customer makes purchases directly impacts their likelihood of continuing to buy.
- **Time Since Last Purchase**: A longer time since the last purchase can indicate a higher risk of churn.
- **Amount Spent**: Customers who spend less may be at higher risk of churning.
- **Age**: Certain age demographics may exhibit different purchasing behaviors.
- **Customer Service Interactions**: More interactions with customer service could indicate dissatisfaction.

### Suggestions for Interpreting Predictions

- **Probability Scores**: The model outputs probability scores for each customer. A score closer to 1 indicates a higher likelihood of churn.
- **Threshold Setting**: Depending on the business needs, a threshold can be set to classify customers as likely to churn or not. For example, a threshold of 0.5 might classify any score above 0.5 as churn.
- **Actionable Insights**: Businesses can use these insights to develop targeted marketing strategies for customers at high risk of churn.

---

### 3. Sample Predictions

Here’s an example of how to generate sample predictions using the trained model:

```python
# Sample new customer data (make sure it matches your feature structure)
sample_data = pd.DataFrame({
    'Purchase_Frequency': [5, 1, 10],
    'Time_Since_Last_Purchase': [30, 200, 10],
    'Amount_Spent': [150, 30, 200],
    'Age': [25, 45, 30],
    'Customer_Service_Interactions': [1, 5, 0]
})

# Make predictions using the trained model
predictions = model.predict(sample_data)
prediction_probabilities = model.predict_proba(sample_data)[:, 1]  # Probability of churn

# Create a DataFrame for results
results = pd.DataFrame({
    'Predicted_Churn': predictions,
    'Churn_Probability': prediction_probabilities
})

print(results)

````
