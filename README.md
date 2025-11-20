# 🏠 Loan Eligibility Status Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Business Problem
A Housing Finance Company deals in all kinds of home loans across urban, semi-urban, and rural areas. Currently, the loan eligibility process is manual and time-consuming.

The objective of this project is to **automate the loan eligibility process (real-time)** based on customer details provided while filling out online application forms. By analyzing historical data, we identify customer segments that are eligible for loan amounts, allowing the company to specifically target these customers.

## 📊 Dataset Details
The project uses the `LoanApprovalPrediction.csv` dataset, which contains 13 columns and 598 records.
* **Target Variable:** `Loan_Status` (Y = Approved, N = Rejected)
* **Key Features:**
    * `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
    * `ApplicantIncome`, `CoapplicantIncome` (Combined into `Income` during preprocessing)
    * `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
    * `Property_Area` (Urban, Semi-Urban, Rural)

## 🛠️ Project Workflow

### 1. Data Cleaning & Preprocessing
* **Missing Value Treatment:**
    * `Dependents`: Filled missing values with `0`.
    * `Income`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`: Dropped rows with null values to ensure data quality.
* **Feature Engineering:**
    * Created a new feature **`Income`** by summing `ApplicantIncome` and `CoapplicantIncome`.
    * Dropped original `ApplicantIncome`, `CoapplicantIncome`, and `Loan_ID` columns.
* **Transformations:**
    * Applied **Box-Cox Transformation** to `Income` and `LoanAmount` to handle skewness and normalize distribution.
* **Encoding:**
    * Manual encoding applied to categorical variables (e.g., `Male:1`, `Female:0`, `Urban:2`, `Semiurban:1`, `Rural:0`).

### 2. Exploratory Data Analysis (EDA)
We analyzed the distribution of data and relationships between features.
* **Univariate Analysis:** Histograms and Boxplots used to identify skewness and outliers.
* **Bivariate Analysis:** Correlation Heatmaps and Pairplots.
* **Key Insight:** `Credit_History` has the highest positive correlation with `Loan_Status`.

### 3. Model Building
The dataset was split into **80% Training** and **20% Testing** sets. The following algorithms were implemented and tuned using **GridSearchCV**:

1.  Logistic Regression
2.  K-Nearest Neighbors (KNN)
3.  Decision Tree Classifier
4.  Random Forest Classifier
5.  AdaBoost Classifier
6.  Gradient Boosting Classifier
7.  XGBoost Classifier

## 📈 Model Performance
After Hyperparameter Tuning (HPT) and cross-validation, here are the accuracy scores on the Test set:

| Model | Test Accuracy | Key Findings |
| :--- | :---: | :--- |
| **Logistic Regression** | **81.55%** | Robust and simple. Selected as the final model. |
| XGBoost | 79.84% | Good performance, but slightly overfitted compared to Logistic. |
| Gradient Boosting | 79.61% | Strong predictor, identifying `Credit_History` as a key feature. |
| Decision Tree | 78.64% | Pruned with max_depth=4. |
| Random Forest | 77.66% | Tuned with n_estimators=24. |
| KNN | 73.78% | Performed poorly compared to tree-based models. |

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/samanthuday0339/Loan_Apporval_prediction.git](https://github.com/samanthuday0339/Loan_Apporval_prediction.git)
    cd Loan_Apporval_prediction
    ```

2.  **Install necessary libraries**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy joblib
    ```

3.  **Run the Notebook/Script**
    Execute the main script or open the Jupyter Notebook to see the training process.

## 💾 Model Saving
The best performing model (**Logistic Regression**) was saved using `joblib` for future deployment:
```python
from joblib import dump
dump(log_model, 'loan_predict_model')
